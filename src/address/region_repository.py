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
