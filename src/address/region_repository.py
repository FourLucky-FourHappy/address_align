import os
from functools import cached_property

from sqlalchemy import URL, create_engine, text
from sqlalchemy.engine import Engine


_SELECT_REGION = (
    "SELECT id, name, statis_code, code, full_name, region_type, parent_id FROM region"
)


class RegionRepository:
    """查询 MySQL 中的行政区划数据。"""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3306,
        user: str = "root",
        database: str = "address_project",
    ):
        self.host = host
        self.port = port
        self.user = user
        self.database = database

    def find_by_name(self, name: str, region_type: int | None = None) -> list[dict]:
        """精确查找地区名称，保留重名的候选项。"""
        sql = f"{_SELECT_REGION} WHERE name = :name"
        params: dict[str, str | int] = {"name": name}
        if region_type is not None:
            sql += " AND region_type = :region_type"
            params["region_type"] = region_type

        return self._fetch_all(sql + " ORDER BY id", params)

    def get_by_id(self, region_id: int) -> dict | None:
        """按地区 ID 查找记录，可传入子地区的 parent_id 查上级。"""
        matches = self._fetch_all(f"{_SELECT_REGION} WHERE id = :id", {"id": region_id})
        return matches[0] if matches else None

    def list_children(self, parent_id: int, region_type: int | None = None) -> list[dict]:
        """查询指定地区的直接下级，可按地区类型缩小范围。"""
        sql = f"{_SELECT_REGION} WHERE parent_id = :parent_id"
        params: dict[str, str | int] = {"parent_id": parent_id}
        if region_type is not None:
            sql += " AND region_type = :region_type"
            params["region_type"] = region_type

        return self._fetch_all(sql + " ORDER BY id", params)

    def search_candidates(self, queries: list[tuple[int, str]]) -> list[dict]:
        """一次查出多组（地区类型、名称片段）的候选。"""
        terms = dict.fromkeys((level, name.strip()) for level, name in queries if name.strip())
        if not terms:
            return []

        conditions = []
        params: dict[str, str | int] = {}
        for index, (level, name) in enumerate(terms):
            level_param = f"level_{index}"
            name_param = f"name_{index}"
            conditions.append(f"(region_type = :{level_param} AND name LIKE :{name_param} ESCAPE '!')")

            # 用户输入中的 %、_ 和 ! 作为普通字符参与查询。
            escaped = name.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            params[level_param] = level
            params[name_param] = f"%{escaped}%"

        sql = f"{_SELECT_REGION} WHERE " + " OR ".join(conditions) + " ORDER BY id"
        return self._fetch_all(sql, params)

    @cached_property
    def _engine(self) -> Engine:
        # 第一次查询时创建引擎，后续查询复用同一个连接池。
        password = os.environ.get("ADDRESS_DB_PASSWORD")
        if password is None:
            raise RuntimeError("请先设置环境变量 ADDRESS_DB_PASSWORD")

        url = URL.create(
            "mysql+pymysql",
            username=self.user,
            password=password,
            host=self.host,
            port=self.port,
            database=self.database,
            query={"charset": "utf8mb4"},
        )
        return create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 5})

    def _fetch_all(self, sql: str, params: dict[str, str | int]) -> list[dict]:
        # 查询值使用命名参数绑定，退出 with 时连接归还到连接池。
        with self._engine.connect() as connection:
            rows = connection.execute(text(sql), params).mappings()
            return [dict(row) for row in rows]


if __name__ == "__main__":
    repository = RegionRepository()

    # 先按名称查城市，再用城市记录里的 ID 查上级和下级。
    cities = repository.find_by_name("深圳市", region_type=3)
    if cities:
        city = cities[0]
        print(f"城市：{city['name']}（ID：{city['id']}）")

        province = repository.get_by_id(city["parent_id"])
        print(f"上级地区：{province['name'] if province else '无'}")

        districts = repository.list_children(city["id"], region_type=4)
        district_names = "、".join(district["name"] for district in districts if district["name"])
        print(f"下级区县：{district_names or '无'}")
    else:
        print("未找到深圳市")

    # 同一个区名可能属于不同城市，完整名称可以用来区分。
    print("同名区县「南山区」的候选：")
    for district in repository.find_by_name("南山区", region_type=4):
        print(f"- {district['full_name']}")
