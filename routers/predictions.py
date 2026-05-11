"""
routers/predictions.py
──────────────────────
FastAPI router để log các lần dự đoán từ frontend:
  POST /predictions/log  — ghi nhận 1 lần dự đoán (guest hoặc user)
  GET  /predictions/my   — lịch sử dự đoán của user hiện tại
"""
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database.database import get_db
from database.models import ActivityLog, PredictionLog, User
from routers.auth import get_current_user, _get_ip

router = APIRouter(prefix="/predictions", tags=["predictions"])


class PredictionLogRequest(BaseModel):
    drug:         Optional[str] = None
    disease:      Optional[str] = None
    dataset:      Optional[str] = None
    type:         Optional[str] = "single"
    top_k:        Optional[int] = 10
    result_count: Optional[int] = 0


@router.post("/log")
async def log_prediction(
    body: PredictionLogRequest,
    request: Request,
    current_user: Optional[User] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Ghi nhận 1 lần dự đoán. Chấp nhận cả guest (user_id=None) và user đã login.
    Frontend gọi endpoint này sau mỗi lần predict thành công.
    """
    try:
        ip = _get_ip(request)
        user_id = current_user.id if current_user else None

        log = PredictionLog(
            user_id         = user_id,
            drug_name       = body.drug,
            disease_name    = body.disease,
            dataset         = body.dataset,
            top_k           = body.top_k,
            result_count    = body.result_count,
            prediction_type = body.type,
            ip_address      = ip,
        )
        db.add(log)

        # Also write to activity log
        db.add(ActivityLog(
            user_id    = user_id,
            action     = "PREDICT",
            detail     = json.dumps({
                "drug": body.drug, "disease": body.disease,
                "dataset": body.dataset, "type": body.type,
            }),
            ip_address = ip,
        ))
        db.commit()
        return {"message": "Logged", "log_id": log.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/my")
async def my_predictions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 50,
):
    """Lịch sử dự đoán của user đang đăng nhập."""
    if not current_user:
        raise HTTPException(status_code=401, detail="Cần đăng nhập")
    logs = (
        db.query(PredictionLog)
        .filter(PredictionLog.user_id == current_user.id)
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return {"logs": [l.to_dict() for l in logs]}
