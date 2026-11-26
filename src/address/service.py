import re
from typing import overload

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from address.extract import extract_address
from address.region_repository import RegionRepository


_REGION_TYPES = {"prov": 2, "city": 3, "district": 4, "town": 5}
_ALTERNATE_TYPES = (("city", 2), ("prov", 3), ("district", 3), ("city", 4))
_MOBILE_PHONE = re.compile(r"1[0-9]{10}")
_LANDLINE_PHONE = re.compile(
    r"(?:\+?[0-9]{1,4}[-.\s]?)?(?:(?:\([0-9]{2,4}\)|[0-9]{2,4})[-.\s]?)?[0-9]{6,11}"
)
_AddressFields = dict[str, str | None]


class AddressAlignmentService:
    """把地址文本提取为字段，再用行政区划候选校验和补全。"""

    def __init__(self, predictor, repository: RegionRepository):
        self.predictor = predictor
        self.repository = repository

    @overload
    def align_address(self, text: str) -> _AddressFields: ...

    @overload
    def align_address(self, text: list[str]) -> list[_AddressFields]: ...

    def align_address(self, text: str | list[str]) -> _AddressFields | list[_AddressFields]:
        """对齐单条或多条地址，批量结果按输入顺序返回。"""
        if isinstance(text, str):
            return self._align_one(text, self.predictor.predict(text))
        if not text:
            return []

        # 整批只调用一次模型预测，再逐条提取字段并查询地区表。
        batch_labels = self.predictor.predict(text)
        return [self._align_one(address, labels) for address, labels in zip(text, batch_labels, strict=True)]

    def _align_one(self, text: str, labels: list[str]) -> _AddressFields:
        fields = extract_address(text, labels)
        phone = fields["phone"]
        if phone and not (_MOBILE_PHONE.fullmatch(phone) or _LANDLINE_PHONE.fullmatch(phone)):
            fields["phone"] = None

        hints = {}
        for field in _REGION_TYPES:
            name = fields[field]
            if name and len(name.strip()) > 1:
                hints[field] = name.strip()
        if not hints:
            return fields

        queries = [(_REGION_TYPES[field], name) for field, name in hints.items()]
        # 例如模型可能把“北京市”标成 city，把县级市标成地级市。
        queries.extend((level, hints[field]) for field, level in _ALTERNATE_TYPES if field in hints)
        match = self._select_candidate(text, hints, self.repository.search_candidates(queries))
        if match is None:
            return fields

        chain, parent = match
        aligned = dict(zip(_REGION_TYPES, chain))
        for field, level in _REGION_TYPES.items():
            if field in aligned:
                continue
            if field not in hints:
                break

            children = self.repository.list_children(parent["id"], region_type=level)
            child = self._match_child(hints[field], children)
            if child is None:
                break
            aligned[field] = child["name"]
            parent = child

        # 确认一条链后，以地区表的名称为准，清除这条链上无法确认的下级。
        fields.update({field: aligned.get(field) for field in _REGION_TYPES})
        return fields

    @staticmethod
    def _select_candidate(
        text: str, hints: dict[str, str], candidates: list[dict],
    ) -> tuple[tuple[str, ...], dict] | None:
        paths = {}
        for region in candidates:
            chain = tuple((region["full_name"] or "").split())
            if len(chain) == region["region_type"] - 1 and 1 <= len(chain) <= len(_REGION_TYPES):
                paths.setdefault(chain, region)
        if not paths:
            return None

        terms = set(hints.values())
        observed = "".join(hints.values())
        scores = {}
        for chain in paths:
            exact = frozenset(terms.intersection(chain))
            partial_count = sum(any(term in name for name in chain) for term in terms)
            text_count = sum(name in text for name in chain)
            # 只比较有原始信息的层级，避免缺失的省市名称拉低相似度。
            comparable = "".join(
                name for field, name in zip(_REGION_TYPES, chain) if field in hints or name in text
            )
            scores[chain] = ((len(exact), partial_count, text_count), exact, fuzz.ratio(comparable, observed))

        # 优先比较命中的地址线索，再用字符串相似度排序。
        best_evidence = max(score[0] for score in scores.values())
        finalists = [chain for chain, score in scores.items() if score[0] == best_evidence]
        if len({scores[chain][1] for chain in finalists}) > 1:
            # 两条链分别吻合相互冲突的线索，且证据一样多时保留原结果。
            return None

        best_similarity = max(scores[chain][2] for chain in finalists)
        winners = [chain for chain in finalists if scores[chain][2] == best_similarity]
        shortest = min(winners, key=len)
        if all(chain[:len(shortest)] == shortest for chain in winners):
            # 同一条链上分数相同的前缀只取能确认的层级，不猜更细的地区。
            return shortest, paths[shortest]
        return None

    @staticmethod
    def _match_child(name: str, candidates: list[dict]) -> dict | None:
        exact = [region for region in candidates if region["name"] == name]
        if exact:
            return exact[0] if len(exact) == 1 else None

        # 在已知上级的范围内，只接受一处增删或替换且结果唯一的地名。
        close = [
            region for region in candidates
            if region["name"] and min(len(name), len(region["name"])) >= 3
            and Levenshtein.distance(name, region["name"], score_cutoff=1) <= 1
        ]
        return close[0] if len(close) == 1 else None
