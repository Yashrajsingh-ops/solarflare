"""
upload.py — CSV ingestion endpoint.
"""

import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.models.database import SolarObservation, UserActivity, get_db
from app.schemas.schemas import UploadResponse
from app.services.preprocessing import parse_and_validate_csv

router = APIRouter(prefix="/api", tags=["Upload"])
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=UploadResponse, summary="Upload SOLEXS / HEL1OS CSV")
async def upload_dataset(
    file: UploadFile = File(..., description="CSV with timestamp, soft_xray_flux, hard_xray_flux"),
    db: Session = Depends(get_db),
) -> UploadResponse:
    """
    Accept a CSV file and persist valid observations to the database.

    The file must contain at minimum the columns:
    `timestamp`, `soft_xray_flux`, `hard_xray_flux`.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    content = await file.read()

    try:
        df = parse_and_validate_csv(content)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    rows = [
        SolarObservation(
            timestamp=row["timestamp"],
            soft_xray_flux=row["soft_xray_flux"],
            hard_xray_flux=row["hard_xray_flux"],
        )
        for _, row in df.iterrows()
    ]

    db.bulk_save_objects(rows)
    db.add(
        UserActivity(
            action="upload",
            entity_type="dataset",
            entity_id=file.filename,
            details_json=f'{{"rows_ingested": {len(rows)}}}',
        )
    )
    db.commit()

    logger.info("Ingested %d observations from '%s'", len(rows), file.filename)
    return UploadResponse(message="Dataset uploaded successfully", rows_ingested=len(rows))
