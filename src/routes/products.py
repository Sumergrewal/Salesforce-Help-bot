from fastapi import APIRouter
from src.agent.db import fetchall
from src.models.schemas import ProductsResponse

router = APIRouter(prefix="/products", tags=["products"])

@router.get("", response_model=ProductsResponse)
def list_products():
    rows = fetchall(
        """
        SELECT DISTINCT product
        FROM fsc_chunks
        WHERE product IS NOT NULL AND product <> ''
        ORDER BY product ASC;
        """
    )
    products = [r["product"] for r in rows]
    return ProductsResponse(products=products)
