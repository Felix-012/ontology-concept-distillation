import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import faiss
import numpy as np
import pandas as pd
import torch
from rich import print as rprint
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from radgraph import RadGraph

# constants for from_default_constants()
SAPBERT_MODEL_ID: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
INDEX_DIR: Path = Path("/vol/ideadata/ce90tate/data/faiss/sapbert_umls_index")
INDEX_FILE: Path = INDEX_DIR / "sapbert.index"
MAPPING_FILE: Path = INDEX_DIR / "sapbert_id2cui.json"
RRF_FILE: Path = Path("/vol/ideadata/ce90tate/data/umls/2024AB/META/MRCONSO.RRF")

K_CANDIDATES: int = 40
BATCHSIZE_EMBED: int = 128
BATCHSIZE_LINK: int = 256
USE_GPU: bool = torch.cuda.is_available()
DTYPE = torch.float16 if USE_GPU else torch.float32
DEVICE=3

@dataclass
class Span:
    """Character‑level span in original text (approx.)."""

    start: int
    end: int

    def to_tuple(self) -> Tuple[int, int]:
        return self.start, self.end


@dataclass
class Mention:
    """A detected entity mention with SapBERT‑linked CUI."""

    text: str
    span: Span
    category: str  # ANATOMY / OBSERVATION …
    assertion: str  # present | negated | uncertain | na
    cui: Optional[str] = None
    score: Optional[float] = None

    def to_json(self) -> Dict:
        d = asdict(self)
        d["span"] = self.span.to_tuple()
        return d


class ClinicalEntityLinker:

    def __init__(self,
                 rrf_file,
                 sapbert_model_id,
                 mapping_file,
                 index_dir,
                 index_file,
                 dtype=torch.float16,
                 device=0):

        self.rrf_file = rrf_file
        self.index_dir = index_dir
        self.index_file = index_file

        rprint("[bold cyan]Loading RadGraph…[/bold cyan]")
        self.radgraph = RadGraph(cuda=device)
        rprint("[bold cyan]Loading SapBERT encoder…[/bold cyan]")
        self.sapbert = AutoModel.from_pretrained(
            sapbert_model_id, torch_dtype=dtype
        ).to(f"cuda:{device}")
        self.sapbert_tokenizer = AutoTokenizer.from_pretrained(sapbert_model_id)

        if not Path(self.index_file).exists():
            rprint(
                f"[yellow]FAISS index not found in {index_dir}. Building from UMLS synonyms…[/yellow]"
            )
            self._build_faiss_index()

        rprint("[bold cyan]Loading FAISS index…[/bold cyan]")
        self.index = faiss.read_index(str(index_file))
        res = faiss.StandardGpuResources()
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        co.indicesOptions = faiss.INDICES_32_BIT
        rprint("[bold cyan]Moving index to GPU…[/bold cyan]")
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index, co)
        with open(mapping_file, "r", encoding="utf‑8") as fp:
            self.id2cui: Dict[str, str] = json.load(fp)

    @classmethod
    def from_default_constants(cls):
        return cls(
            rrf_file=RRF_FILE,
            sapbert_model_id=SAPBERT_MODEL_ID,
            mapping_file=MAPPING_FILE,
            index_dir=INDEX_DIR,
            index_file=INDEX_FILE,
            dtype=DTYPE,
            device=DEVICE
        )


    def __call__(self, note: str) -> List[Mention]:
        mentions = self._infer_ner(note)
        self._link_mentions_batch(mentions)
        return mentions


    def _infer_ner(self, note: str) -> List[Mention]:
        """Run RadGraph and convert its output to List[Mention]."""
        ann = self.radgraph([note])
        # Grab first (and only) doc
        ann_doc = next(iter(ann.values())) if isinstance(ann, dict) else ann[0]
        entities = ann_doc["entities"]

        mentions: List[Mention] = []
        for ent in entities.values():
            label = ent["label"]
            mention_text = ent["tokens"]
            relations = ent["relations"]
            if label.endswith("definitely absent"):
                for relation in relations:
                    entities[relation[1]]["label"] = entities[relation[1]]["label"].replace("definitely present", "definitely absent")


            # Approximate char indices (token‑level indices available in ent)
            char_start = note.lower().find(mention_text.lower())
            char_end = char_start + len(mention_text) if char_start != -1 else -1

            assertion = {
                'Observation::definitely present': "present",
                'Observation::definitely absent': "absent",
                'Observation::uncertain': "uncertain",
                'Anatomy::definitely present': "present",
                'Anatomy::definitely absent': "absent",
                'Anatomy::uncertain': "uncertain"
            }.get(label, "na")

            clean_cat = label.replace("NEGATED_", "").replace("UNCERTAIN_", "")

            mentions.append(
                Mention(
                    text=mention_text,
                    span=Span(char_start, char_end),
                    category=clean_cat,
                    assertion=assertion,
                )
            )
        return mentions


    def _link_mentions_batch(self, mentions: List[Mention], batchsize_link: int = BATCHSIZE_LINK):
        if not mentions:
            return
        # Deduplicate identical surface strings to minimise encoder work
        unique_texts: List[str] = []
        text_to_idx: Dict[str, int] = {}
        for m in mentions:
            if m.text not in text_to_idx:
                text_to_idx[m.text] = len(unique_texts)
                unique_texts.append(m.text)

        all_vecs = []
        for i in range(0, len(unique_texts), batchsize_link):
            batch_vec = self._encode_text(unique_texts[i : i + batchsize_link])
            all_vecs.append(batch_vec)
        vecs = np.vstack(all_vecs)
        sims, idxs = self.index.search(vecs, 1)

        # Map back to each mention
        for m in mentions:
            uidx = text_to_idx[m.text]
            cui = self.id2cui[str(int(idxs[uidx][0]))]
            m.cui = cui
            m.score = float(sims[uidx][0])


    @torch.no_grad()
    def _encode_text(self, text: Union[str, List[str]]) -> np.ndarray:
        toks = self.sapbert_tokenizer(
            text,
            max_length=25,
            truncation=True,
            return_tensors="pt",
            padding="max_length",
        )
        toks = {k: v.to(self.sapbert.device) for k, v in toks.items()}
        vec = self.sapbert(**toks)[0][:, 0, :]  # CLS token
        vec = torch.nn.functional.normalize(vec, dim=1)
        return vec.cpu().numpy().astype("float32")

    def _sapbert_link(self, mention: str) -> Tuple[str, float]:
        vec = self._encode_text(mention)
        sims, idxs = self.index.search(vec, 1)
        cui = self.id2cui[str(int(idxs[0][0]))]
        return cui, float(sims[0][0])

    def _load_umls_synonyms(self, rrf_path: Union[str, Path]) -> pd.DataFrame:
        rrf_path = Path(rrf_path)
        compression = "gzip" if rrf_path.suffix == ".gz" else None
        col_names = [
            "CUI",
            "LAT",
            "TS",
            "LUI",
            "STT",
            "SUI",
            "ISPREF",
            "AUI",
            "SAUI",
            "SCUI",
            "SDUI",
            "SAB",
            "TTY",
            "CODE",
            "STR",
            "SRL",
            "SUPPRESS",
            "CVF",
            "NaN",
        ]
        df = pd.read_csv(
            rrf_path,
            sep="|",
            header=None,
            names=col_names,
            usecols=["CUI", "LAT", "TS", "STR"],
            dtype=str,
            compression=compression,
        )
        df = df.loc[(df["LAT"] == "ENG") & (df["TS"] != "S"), ["CUI", "STR"]]
        df.dropna(subset=["CUI", "STR"], inplace=True)
        return df.reset_index(drop=True)

    def _build_faiss_index(self, batch_size_embed: int =BATCHSIZE_EMBED):
        self.index_dir.mkdir()
        syn_df = self._load_umls_synonyms(self.rrf_file)
        id2cui: Dict[int, str] = {}
        names: List[str] = []
        for cui, name in syn_df.itertuples(index=False):
            id2cui[len(names)] = cui
            names.append(name)

        vectors: List[np.ndarray] = []
        for i in range(0, len(names), batch_size_embed):
            batch = names[i : i + batch_size_embed]
            vec = self._encode_text(batch)
            vectors.append(vec)
        mat = np.concatenate(vectors, axis=0)

        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        faiss.write_index(index, str(self.index_file))
        with open(MAPPING_FILE, "w", encoding="utf‑8") as fp:
            json.dump({str(k): v for k, v in id2cui.items()}, fp)
        rprint(f"[green]Built FAISS index with {len(mat):,} vectors.[/green]")


def get_mentions(path: Union[Path, str], impressions: List[str]) -> List[List[Mention]]:
    if not Path.exists(Path(path)):

        linker = ClinicalEntityLinker.from_default_constants()

        mentions = []

        for impression in tqdm(impressions, desc="Linking clinical entities"):
            mentions.append(linker(impression))

        with open(path, "wb") as f:
            pickle.dump(mentions, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved {len(mentions)} mentions to {path}")

    else:
        mentions = pickle.load(open(path, "rb"))

    return mentions