from __future__ import annotations
import gffutils

def open_gff_db(gff_path: str, db_path: str | None = None) -> gffutils.FeatureDB:
    if db_path is None:
        db_path = gff_path + ".db"
    try:
        return gffutils.FeatureDB(db_path)
    except Exception:
        return gffutils.create_db(gff_path, dbfn=db_path, force=True, keep_order=True, merge_strategy="merge")
