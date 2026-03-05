"""PDF 문서를 미리 인덱싱하는 CLI 스크립트.

사용법:
    python ingest.py data/문서1.pdf data/문서2.pdf
    python ingest.py data/           # data/ 폴더 내 모든 PDF
"""
import sys
from pathlib import Path
from src.graph import run_ingestion_pipeline


def main():
    if len(sys.argv) < 2:
        print("사용법: python ingest.py <pdf_path_or_dir> [...]")
        sys.exit(1)

    pdf_paths = []
    for arg in sys.argv[1:]:
        p = Path(arg)
        if p.is_dir():
            pdf_paths.extend(str(f) for f in p.glob("*.pdf"))
        elif p.is_file() and p.suffix.lower() == ".pdf":
            pdf_paths.append(str(p))
        else:
            print(f"건너뜀: {arg} (PDF 파일이 아님)")

    if not pdf_paths:
        print("인덱싱할 PDF 파일이 없습니다.")
        sys.exit(1)

    print(f"총 {len(pdf_paths)}개 PDF 인덱싱 시작:")
    for p in pdf_paths:
        print(f"  - {p}")

    result = run_ingestion_pipeline(pdf_paths)

    if result["status"] == "success":
        print(f"\n인덱싱 완료!")
        print(f"  - 생성된 청크: {result['num_chunks']}개")
        print(f"  - 추출된 엔티티: {result['num_entities']}개")
    else:
        print(f"\n인덱싱 실패: {result}")
        sys.exit(1)


if __name__ == "__main__":
    main()
