from itertools import groupby

from configuration import config


def extract_address(text: str, labels: list[str]) -> dict[str, str | None]:
    """提取连续实体片段，同类字段重复出现时取最后一段。"""
    address: dict[str, str | None] = dict.fromkeys(config.FIELD_NAMES)

    # B-prov、I-prov、E-prov 都属于 prov。O 会得到空字段，分隔前后实体。
    pairs = zip(text, labels, strict=True)
    for field, span in groupby(pairs, key=lambda pair: pair[1].partition("-")[2]):
        if field in address:
            address[field] = "".join(char for char, _ in span)

    return address
