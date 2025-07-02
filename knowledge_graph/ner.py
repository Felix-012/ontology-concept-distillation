import json

import spacy
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import faiss
import numpy as np
import pandas as pd
import torch
import yaml
from rich import print as rprint
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from radgraph import RadGraph

SAPBERT_MODEL_ID: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
#SAPBERT_MODEL_ID: str = "microsoft/BiomedNLP-KRISSBERT-PubMed-UMLS-EL"

K_CANDIDATES: int = 40
BATCHSIZE_EMBED: int = 128
BATCHSIZE_LINK: int = 256
USE_GPU: bool = torch.cuda.is_available()
DTYPE = torch.float16 if USE_GPU else torch.float32
DEVICE=0

LABELS = ["Atelectasis",
          "Cardiomegaly",
          "Consolidation",
          "Edema",
          "Effusion",
          "Pneumonia",
          "Pneumothorax",
          "Opacity",
          "Device",
          "Fracture",
          "Pleural",
          "Lesion"
          ]


CUI_TO_LABEL = {"C0004144": "Atelectasis",
                "C0018800": "Cardiomegaly",
                "C0521530": "Consolidation",
                "C0013604": "Pleural Effusion",
                "C0032227": "Pneumonia",
                "C0032285": "Pneumothorax",
                "C5921295": "Opacity",
                "C0025080": "Device",
                "C0016658": "Fracture",
                frozenset(): "No Finding"
                }

LABEL_TO_CUI = {
    "Atelectasis": {"C0004144"},
    "Cardiomegaly": {"C0018800"},
    "Enlarged Cardiomediastinum": {"C0018800"},
    "Consolidation": {"C0521530"},
    "Edema": {"C0013604"},
    "Pleural Effusion": {'C0032227'},
    "Pleural Other": {"C0348709"},
    "Pneumonia": {"C0032285"},
    "Pneumothorax": {"C0032326"},
    "Lung Opacity": {"C5921295"},
    "Support Devices": {"C0025080", },
    "Fracture": {"C0016658"},
    "Lung Lesion": {"C0577916"},
    "No Finding": frozenset()
}

PRECEDENCE = {"present": 2, "uncertain": 1, "absent": 3}

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
    assertion: str # present | negated | uncertain | na
    mods: List[str]
    cui: Optional[str] = None
    cui_text: Optional[str] = None,
    cui_surface: Optional[str] = None,
    score: Optional[float] = None,
    score_text: Optional[str] = None,
    score_surface: Optional[str] = None,


    def to_json(self) -> Dict:
        d = asdict(self)
        d["span"] = self.span.to_tuple()
        return d


class ClinicalEntityLinker:

    def __init__(self,
                 conso_file,
                 sty_file,
                 mrrel_file,
                 sapbert_model_id,
                 mapping_file,
                 index_dir,
                 index_file,
                 dtype=torch.float16,
                 device=DEVICE):

        self.mapping_file = mapping_file
        self.conso_file = conso_file
        self.index_dir = index_dir
        self.index_file = index_file

        rprint("[bold cyan]Loading RadGraph…[/bold cyan]")
        self.radgraph = RadGraph(cuda=device)
        rprint("[bold cyan]Loading SapBERT encoder…[/bold cyan]")
        self.sapbert = AutoModel.from_pretrained(
            sapbert_model_id, torch_dtype=dtype
        ).to(f"cuda:{device}")
        self.sapbert_tokenizer = AutoTokenizer.from_pretrained(sapbert_model_id)
        self.syns = self._load_umls_synonyms(conso_file)
        #self.reranker_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        #self.reranker_model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder")

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
        self.index = faiss.index_cpu_to_gpu(res, DEVICE, self.index, co)
        with open(mapping_file, "r", encoding="utf‑8") as fp:
            self.id2cui: Dict[str, str] = json.load(fp)
        # at module level – build once
        self.cats_to_allowed_sty = {"Observation": {"T047", "T046", "T033", "T019", "T037"},
                                    "Anatomy":     {"T017", "T023", "T029", "T030", "T082"},}
        self.cui2sty = _load_mrsty(sty_file)
        #self.cui2def = _load_mrdef(def_file)
        self.cui2str = dict(self.syns.values)
        self.importance_scores_sty = {"T047": 3, "T046": 2, "T033": 1, "T019": 0,  "T037": 0,
                                      "T017": 0, "T023": 0, "T029": 0, "T030": 0, "T082": 0}
        self.relations = self._load_mrrel(mrrel_file)
        #self.relations = None


    @classmethod
    def from_config_path(cls, config_path):
        with open(config_path, "r") as stream:
            config = yaml.safe_load(stream)
        return cls(
            conso_file=config["mrconso_path"],
            sty_file=config["mrsty_path"],
            mrrel_file=config["mrrel_path"],
            sapbert_model_id=SAPBERT_MODEL_ID,
            mapping_file=config["mapping_file"],
            index_dir=config["index_dir"],
            index_file=config["index_file"],
            dtype=DTYPE,
            device=DEVICE
        )


    def __call__(self, note: str) -> List[Mention]:
        mentions = self._infer_ner(note)
        self._link_mentions_batch(mentions)
        return mentions

    def _infer_ner(self, note: str) -> List[Mention]:
        small_note = len(note.split()) < 3
        if small_note:
            alt_assertion = self._negation_tool(note)

        sentences = [s.text.strip() for s in self._nlp(note).sents]
        ann_sent = self.radgraph(sentences)
        ann_note = self.radgraph([note]) if len(sentences) > 1 else ann_sent

        ann_sent_docs = list(ann_sent.values()) if isinstance(ann_sent, dict) else ann_sent
        ann_note_doc = next(iter(ann_note.values())) if isinstance(ann_note, dict) else ann_note[0]
        note_label_map = {e["tokens"].lower(): e["label"] for e in ann_note_doc["entities"].values()}

        mentions: List[Mention] = []
        for ann_doc in ann_sent_docs:
            entities = ann_doc["entities"]

            for ent in entities.values():
                tok = ent["tokens"].lower()
                if tok in note_label_map:
                    ent["label"] = note_label_map[tok]

            for ent in entities.values():
                if ent["label"].endswith("definitely absent"):
                    for r in ent["relations"]:
                        rel_ent = entities[r[1]]
                        if rel_ent["label"].endswith("definitely present"):
                            rel_has_present = any(
                                entities[rr[1]]["label"].endswith("definitely present")
                                for rr in rel_ent["relations"]
                            )
                            if not rel_has_present:
                                rel_ent["label"] = rel_ent["label"].replace(
                                    "definitely present", "definitely absent"
                                )

            for ent in entities.values():
                label = ent["label"]
                mention_text = ent["tokens"]
                relations = ent["relations"]
                mods = [
                    entities[r[1]]["tokens"]
                    for r in relations
                    if entities[r[1]]["label"].startswith("Anatomy")
                ]

                char_start = note.lower().find(mention_text.lower())
                char_end = char_start + len(mention_text) if char_start != -1 else -1

                assertion = {
                    "Observation::definitely present": "present",
                    "Observation::definitely absent": "absent",
                    "Observation::uncertain": "uncertain",
                    "Anatomy::definitely present": "present",
                    "Anatomy::definitely absent": "absent",
                    "Anatomy::uncertain": "uncertain",
                }.get(label, "na")

                if small_note and alt_assertion:
                    if alt_assertion in {"present", "absent", "uncertain"} and alt_assertion != assertion:
                        print(f"{alt_assertion}: {note}")
                        assertion = alt_assertion

                mentions.append(
                    Mention(
                        text=mention_text,
                        span=Span(char_start, char_end),
                        category=label,
                        assertion=assertion,
                        mods=mods,
                    )
                )
        return mentions

    def _link_mentions_batch(
            self,
            mentions: List[Mention],
            batchsize_link: int = BATCHSIZE_LINK,
            top_k: int = 128,
    ) -> None:
        if not mentions:
            return

        surf_strings: List[str] = []  # global list → ANN query order
        string_to_idx: Dict[str, int] = {}  # cache to avoid duplicates


        for m in mentions:
            mods = getattr(m, "mods", None) or []
            surf_str = " ".join(mods + [m.text]).strip()  # “surface”
            text_str = m.text.strip()  # plain text

            # register / deduplicate both strings
            for s in (surf_str, text_str):
                if s not in string_to_idx:
                    string_to_idx[s] = len(surf_strings)
                    surf_strings.append(s)

            # remember the row indices for later look-up
            m._surface_idx = string_to_idx[surf_str]
            m._text_idx = string_to_idx[text_str]

        vecs = np.vstack(
            [
                self._encode_text(surf_strings[i: i + batchsize_link])
                for i in range(0, len(surf_strings), batchsize_link)
            ]
        )
        sims, idxs = self.index.search(vecs, top_k)
        cuis_per_row = [
            [self.id2cui[str(int(idx))] for idx in row] for row in idxs
        ]

        for m in mentions:
            allowed_stys = self.cats_to_allowed_sty.get(
                m.category.split("::")[0], None
            )

            def _select(row_idx: int) -> tuple[str, float]:
                """Return (cui, score) for a single ANN result row."""
                cand_cuis, cand_sims, cand_idxs = (
                    cuis_per_row[row_idx],
                    sims[row_idx],
                    idxs[row_idx],
                )

                # take the first CUI whose semantic-type is allowed
                for rank, cui in enumerate(cand_cuis):
                    if (allowed_stys is None) or (self.cui2sty.get(cui) in allowed_stys):
                        return (
                            self.id2cui[str(int(cand_idxs[rank]))],
                            float(cand_sims[rank]),
                        )

                # fallback: very top ANN hit
                return (
                    self.id2cui[str(int(cand_idxs[0]))],
                    float(cand_sims[0]),
                )

            # surface-based link
            m.cui_surface, m.score_surface = _select(m._surface_idx)

            # plain-text link
            m.cui_text, m.score_text = _select(m._text_idx)

            m.cui = m.cui_surface
            m.score = m.score_surface

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
            usecols=["CUI", "LAT", "TS", "SAB", "STR"],
            dtype=str,
            compression=compression,
        )
        df = df.loc[(df["LAT"] == "ENG") & (df["TS"] != "S") & (df["SAB"] == "SNOMEDCT_US"), ["CUI", "STR"]]
        df.dropna(subset=["CUI", "STR"], inplace=True)
        return df.reset_index(drop=True)

    def _build_faiss_index(self, batch_size_embed: int =BATCHSIZE_EMBED):
        self.index_dir.mkdir()
        syn_df = self.syns
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
        with open(self.mapping_file, "w", encoding="utf‑8") as fp:
            json.dump({str(k): v for k, v in id2cui.items()}, fp)
        rprint(f"[green]Built FAISS index with {len(mat):,} vectors.[/green]")

    def rerank(self, mention, cuis):
        pairs = [[mention, self.cui2str[cui]] for cui in cuis]

        with torch.no_grad():
            encoded = self.reranker_tokenizer(
                pairs,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            )

            logits = self.reranker_model(**encoded).logits.squeeze(dim=1)
            reranked_cuis = [cui for _, cui in sorted(zip(logits, cuis), reverse=True)]
            return reranked_cuis

    def _load_mrrel(self, mrrel_path: Union[str, Path]) -> Dict:
        wanted_rels = {"PAR"}

        rel = pd.read_csv(
            mrrel_path,
            sep="|",
            header=None,
            usecols=[0, 3, 4, 11],
            names=["CUI1", "REL", "CUI2", "SL"],
            dtype="str"
        )

        mask_not_self = rel["CUI1"] != rel["CUI2"]
        mask_wanted_rel = rel["REL"].isin(wanted_rels)
        mask_source = rel["SL"] == "SNOMEDCT_US"
        rel = rel.loc[mask_not_self & mask_wanted_rel & mask_source]

        allowed_sty: set[str] = {"T047", "T046", "T033"}

        def has_allowed_sty(cui: str) -> bool:
            """
            True ⟺ this CUI is mapped to ≥1 allowed semantic type.
            self.cui2sty may map a CUI to:
              • a single STY code  -> str
              • a sequence / set   -> iterable
            """
            sty_codes = self.cui2sty.get(cui, ())
            if isinstance(sty_codes, str):
                sty_codes = (sty_codes,)

            return any(sty in allowed_sty for sty in sty_codes)

        mask_cui1_allowed = rel["CUI1"].apply(has_allowed_sty)
        mask_cui2_allowed = rel["CUI2"].apply(has_allowed_sty)

        rel = rel.loc[mask_cui1_allowed & mask_cui2_allowed]

        rel = rel.drop_duplicates(["CUI1", "CUI2"])
        return rel.groupby("CUI1")['CUI2'].apply(list).to_dict()

    def _negation_tool(self, note: str) -> str:
        """
        Determine whether the given note indicates presence, uncertainty, or absence.
        Expects note to be a short string (more than 3 words).
        Returns one of: "present", "uncertain", "absent".
        """
        words = note.lower().split()

        ABSENCE_KEYWORDS = {"no", "not", "none", "without", "absent"}
        UNCERTAIN_KEYWORDS = {"maybe", "possible", "unclear", "could", "might", "suspect"}

        for kw in UNCERTAIN_KEYWORDS:
            if kw in words:
                return "uncertain"

        for kw in ABSENCE_KEYWORDS:
            if kw in words:
                return "absent"

        return "present"


def get_mentions(linker: ClinicalEntityLinker, path: Union[Path, str], impressions: List[str]) -> List[List[Mention]]:
    if not Path.exists(Path(path)):

        mentions = []

        for impression in tqdm(impressions, desc="Linking clinical entities"):
            mentions.append(linker(impression))

        with open(path, "wb") as f:
            pickle.dump(mentions, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saved {len(mentions)} mentions to {path}")

    else:
        mentions = pickle.load(open(path, "rb"))

    return mentions


def _load_mrsty(mrsty_path: Union[str, Path]) -> Dict[str, str]:
    mrsty_path = Path(mrsty_path)
    compression = "gzip" if mrsty_path.suffix == ".gz" else None
    df = pd.read_csv(
        mrsty_path,
        sep="|",
        header=None,
        names=["CUI","TUI", "STN", "STY", "ATUI", "CVF", "EMPTY"],
        usecols=["CUI", "TUI"],
        dtype=str,
        compression=compression,
    )
    return dict(df.values)


def _load_mrdef(mrsty_path: Union[str, Path]) -> Dict[str, str]:
    mrsty_path = Path(mrsty_path)
    compression = "gzip" if mrsty_path.suffix == ".gz" else None
    df = pd.read_csv(
        mrsty_path,
        sep="|",
        header=None,
        names=["CUI","AUI", "ATUI", "SATUI", "SAB", "DEF", "SUPPRESS", "CVF", "EMPTY"],
        usecols=["CUI", "DEF"],
        dtype=str,
        compression=compression,
    )
    return dict(df.values)



from collections import defaultdict

def create_labels(ids, mentions, linker, reports):
    label_dict = {}
    for id, mentions_per_report, report in zip(ids, mentions, reports):
        label_dict[id] = set()

        for mention in mentions_per_report:

            label_scores = {}

            for label in LABELS:
                score_text = 0
                score_surf = 0

                if label.lower() in linker.cui2str[mention.cui_text].lower():
                    score_text = mention.score_text
                if label.lower() in linker.cui2str[mention.cui_surface].lower():
                    score_surf = mention.score_surface

                score = max(score_text, score_surf)
                if score > 0:  # keep only positive matches
                    label_scores[label] = score

            if label_scores:
                best_label = max(label_scores, key=label_scores.get)
                best_score = label_scores[best_label]
                assertion = mention.assertion

                lower_prec_tuple = None
                for tup in label_dict[id]:
                    same_label = tup[0] == best_label
                    lower_prec = PRECEDENCE[tup[1]] < PRECEDENCE[assertion]
                    if same_label and lower_prec:
                        lower_prec_tuple = tup
                        break

                if lower_prec_tuple:
                    label_dict[id].remove(lower_prec_tuple)

                label_dict[id].add((best_label, assertion))

                if assertion == "present":
                    surf = " ".join(mention.mods + [mention.text]).strip()
                    print(f"{surf}: {best_label} {best_score}")

            if not label_scores:
                for label in LABELS:
                    broader_surf = linker.relations.get(mention.cui_surface)
                    broader_text = linker.relations.get(mention.cui_text)
                    if broader_surf is None:
                        broader_surf = set()
                    else:
                        broader_surf = set(broader_surf)
                    if broader_text is None:
                        broader_text = set()
                    else:
                        broader_text = set(broader_text)
                    broader = broader_surf | broader_text

                    for broad in broader:
                        if label.lower() in linker.cui2str.get(broad, "").lower():
                            for tuple in label_dict[id]:
                                if tuple[0] == label and PRECEDENCE[tuple[1]] < PRECEDENCE[mention.assertion]:
                                    label_dict[id].remove(tuple)
                                    break
                            label_dict[id].add((label, mention.assertion))
                            if mention.assertion == "present":
                                surf = " ".join(mention.mods + [mention.text]).strip()
                                print(f"Broad: {surf}: {label} {mention.score_text} {mention.score_text}")

        if len(label_dict[id]) == 0:
            label_dict[id] = {("No Finding", "present")}
    return label_dict


def jaccard_label(report_cuis, label_cuis):
    sim = len(label_cuis & report_cuis) / len(label_cuis | report_cuis) if (label_cuis | report_cuis) else 0
    return sim

def macro_jaccard(report_cuis, label_cui_sets):
    if not label_cui_sets:
        return 0.0

    sims = []
    for label in label_cui_sets:
        union = report_cuis | label_cui_sets[label]
        sims.append(len(report_cuis & label_cui_sets[label]) / len(union) if union else 0.0)

    return sum(sims) / len(sims)

def coverage_score(report_cuis, label_cuis):
    if not report_cuis:
        return 1.0
    hits = len(report_cuis & label_cuis)
    return hits / len(report_cuis)


def choose_label(mentions, labels1, labels2, linker):
    label1_ctr = label2_ctr = inter_ctr = union_ctr = ident_ctr = 0
    ret_labels = []

    label_dict        = populate_labels(mentions, linker)
    labels1_cuis_all  = _labels_to_cuis(labels1, label_dict)
    labels2_cuis_all  = _labels_to_cuis(labels2, label_dict)

    for m_rpt, lbl1_str, lbl2_str, cuis1_set, cuis2_set in zip(
            mentions, labels1, labels2, labels1_cuis_all, labels2_cuis_all):

        if lbl1_str == lbl2_str:
            ident_ctr += 1
            ret_labels.append(lbl1_str)
            continue

        cui_present = {
            m.cui_text    for m in m_rpt if m.assertion == "present"
        } | {
            m.cui_surface for m in m_rpt if m.assertion == "present"
        }

        lbl1_tokens  = [t for t in lbl1_str.split('|') if t]
        lbl2_tokens  = [t for t in lbl2_str.split('|') if t]
        inter_tokens = [t for t in lbl1_tokens if t in lbl2_tokens]
        union_tokens = list(dict.fromkeys(lbl1_tokens + lbl2_tokens))

        inter_cuis_set = {
            cui
            for tok in inter_tokens
            for cui in label_dict.get(tok, set())
        }
        union_cuis_set = {
            cui
            for tok in union_tokens
            for cui in label_dict.get(tok, set())
        }

        score1    = coverage_score(cui_present, cuis1_set)
        score2    = coverage_score(cui_present, cuis2_set)
        score_int = coverage_score(cui_present, inter_cuis_set) if inter_tokens else -1
        score_uni = coverage_score(cui_present, union_cuis_set)

        best_key, _ = max(
            [
                ("label1",       score1),
                ("label2",       score2),
                ("intersection", score_int),
                ("union",        score_uni),
            ],
            key=lambda x: x[1]
        )

        if best_key == "label1":
            label1_ctr += 1
            ret_labels.append(lbl1_str)

        elif best_key == "label2":
            label2_ctr += 1
            ret_labels.append(lbl2_str)

        elif best_key == "intersection":
            inter_ctr += 1
            ret_labels.append('|'.join(inter_tokens))

        else:  # union
            union_ctr += 1
            ret_labels.append('|'.join(union_tokens))

    print(f"Identical:     {ident_ctr}")
    print(f"Label 1:       {label1_ctr}")
    print(f"Label 2:       {label2_ctr}")
    print(f"Intersection:  {inter_ctr}")
    print(f"Union:         {union_ctr}")

    return ret_labels


def _labels_to_cuis(label_strings, label_dict):
    union_sets = []
    for s in label_strings:
        cuis = set()
        for lab in filter(None, s.split("|")):
            cuis |= label_dict[lab]
        union_sets.append(cuis)
    return union_sets


def populate_labels(mentions, linker: ClinicalEntityLinker):
    label_dict = LABEL_TO_CUI.copy()
    mapping = {"Pleural Effusion": "Effusion",
               "Lung Opacity": "Opacity",}
    for mentions_per_text in mentions:
        for mention in mentions_per_text:
            if linker.cui2sty[mention.cui_surface] == "T074":
                label_dict["Support Devices"].add(mention.cui_surface)
            if  linker.cui2sty[mention.cui_text] == "T074":
                label_dict["Support Devices"].add(mention.cui_text)

            for label in label_dict:
                label_name = mapping.get(label) or label
                label_name = label_name.lower()
                if label_name in linker.cui2str[mention.cui_surface].lower():
                    label_dict[label].add(mention.cui_surface)
                if label_name in linker.cui2str[mention.cui_text].lower():
                    label_dict[label].add(mention.cui_text)

    for label in label_dict:
        label_str = f""
        for cui in label_dict[label]:
            label_str += f" {linker.cui2str.get(cui, '')} "
        print(f"{label}: {label_str}")

    return label_dict






