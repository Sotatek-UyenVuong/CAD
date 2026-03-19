"""
Phân loại PDF files thành Vector PDF và Image-based PDF
"""
import fitz
from pathlib import Path
import json
from datetime import datetime

def analyze_pdf(pdf_path):
    """
    Phân tích PDF để xác định loại
    Returns: dict với thông tin chi tiết
    """
    try:
        doc = fitz.open(pdf_path)
        
        # Thống kê
        total_pages = len(doc)
        total_text_chars = 0
        total_images = 0
        total_paths = 0
        
        # Phân tích từng trang (chỉ check 5 trang đầu để nhanh)
        pages_to_check = min(5, total_pages)
        
        for page_num in range(pages_to_check):
            page = doc[page_num]
            
            # Đếm text
            text = page.get_text()
            total_text_chars += len(text.strip())
            
            # Đếm images
            images = page.get_images()
            total_images += len(images)
            
            # Đếm paths/drawings
            drawings = page.get_drawings()
            total_paths += len(drawings)
        
        # Tính trung bình
        avg_text_per_page = total_text_chars / pages_to_check if pages_to_check > 0 else 0
        avg_images_per_page = total_images / pages_to_check if pages_to_check > 0 else 0
        avg_paths_per_page = total_paths / pages_to_check if pages_to_check > 0 else 0
        
        # Xác định loại PDF
        if avg_images_per_page > 0 and total_text_chars < 100 and avg_paths_per_page < 10:
            pdf_type = "image-based"
            confidence = "high"
        elif total_text_chars > 500 or avg_paths_per_page > 50:
            pdf_type = "vector"
            confidence = "high"
        elif avg_images_per_page > 2:
            pdf_type = "image-based"
            confidence = "medium"
        else:
            pdf_type = "vector"
            confidence = "medium"
        
        # Get file info
        file_size = Path(pdf_path).stat().st_size
        
        result = {
            'filename': Path(pdf_path).name,
            'path': str(pdf_path),
            'type': pdf_type,
            'confidence': confidence,
            'pages': total_pages,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'analysis': {
                'avg_text_chars_per_page': round(avg_text_per_page, 1),
                'avg_images_per_page': round(avg_images_per_page, 1),
                'avg_paths_per_page': round(avg_paths_per_page, 1),
                'pages_analyzed': pages_to_check
            }
        }
        
        doc.close()
        return result
        
    except Exception as e:
        return {
            'filename': Path(pdf_path).name,
            'path': str(pdf_path),
            'type': 'error',
            'error': str(e)
        }

def classify_all_pdfs(pdf_directory, output_file=None):
    """
    Phân loại tất cả PDF trong thư mục
    """
    pdf_dir = Path(pdf_directory)
    pdf_files = list(pdf_dir.glob("*.pdf"))
    
    print("="*70)
    print("🔍 PHÂN LOẠI PDF FILES")
    print("="*70)
    print(f"Thư mục: {pdf_dir}")
    print(f"Tổng số files: {len(pdf_files)}")
    print("\n🔄 Đang phân tích...\n")
    
    results = []
    vector_pdfs = []
    image_pdfs = []
    error_pdfs = []
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"[{idx}/{len(pdf_files)}] {pdf_path.name}...", end=" ")
        
        result = analyze_pdf(pdf_path)
        results.append(result)
        
        if result['type'] == 'vector':
            vector_pdfs.append(result)
            print(f"✅ VECTOR ({result['confidence']})")
        elif result['type'] == 'image-based':
            image_pdfs.append(result)
            print(f"🖼️  IMAGE ({result['confidence']})")
        else:
            error_pdfs.append(result)
            print(f"❌ ERROR")
    
    # Sắp xếp theo size
    vector_pdfs.sort(key=lambda x: x.get('file_size_mb', 0), reverse=True)
    image_pdfs.sort(key=lambda x: x.get('file_size_mb', 0), reverse=True)
    
    # In kết quả
    print("\n" + "="*70)
    print("📊 KẾT QUẢ PHÂN LOẠI")
    print("="*70)
    
    print(f"\n🎯 VECTOR PDFs: {len(vector_pdfs)} files")
    print("-"*70)
    for pdf in vector_pdfs:
        print(f"  • {pdf['filename']}")
        print(f"    {pdf['pages']} pages | {pdf['file_size_mb']} MB | "
              f"Text: {pdf['analysis']['avg_text_chars_per_page']:.0f} chars/page | "
              f"Paths: {pdf['analysis']['avg_paths_per_page']:.0f}/page")
    
    print(f"\n🖼️  IMAGE-BASED PDFs: {len(image_pdfs)} files")
    print("-"*70)
    for pdf in image_pdfs:
        print(f"  • {pdf['filename']}")
        print(f"    {pdf['pages']} pages | {pdf['file_size_mb']} MB | "
              f"Images: {pdf['analysis']['avg_images_per_page']:.1f}/page")
    
    if error_pdfs:
        print(f"\n❌ ERROR PDFs: {len(error_pdfs)} files")
        print("-"*70)
        for pdf in error_pdfs:
            print(f"  • {pdf['filename']}: {pdf.get('error', 'Unknown error')}")
    
    # Tổng kết
    total_vector_size = sum(p.get('file_size_mb', 0) for p in vector_pdfs)
    total_image_size = sum(p.get('file_size_mb', 0) for p in image_pdfs)
    
    print("\n" + "="*70)
    print("📈 THỐNG KÊ TỔNG QUAN")
    print("="*70)
    print(f"✅ Vector PDFs: {len(vector_pdfs)} files ({total_vector_size:.1f} MB)")
    print(f"🖼️  Image PDFs:  {len(image_pdfs)} files ({total_image_size:.1f} MB)")
    print(f"❌ Errors:      {len(error_pdfs)} files")
    print(f"📊 Total:       {len(results)} files ({total_vector_size + total_image_size:.1f} MB)")
    
    # Lưu kết quả ra file JSON
    if output_file:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'directory': str(pdf_dir),
            'summary': {
                'total_files': len(results),
                'vector_pdfs': len(vector_pdfs),
                'image_pdfs': len(image_pdfs),
                'errors': len(error_pdfs),
                'total_size_mb': round(total_vector_size + total_image_size, 2)
            },
            'vector_pdfs': vector_pdfs,
            'image_pdfs': image_pdfs,
            'error_pdfs': error_pdfs
        }
        
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Đã lưu kết quả chi tiết vào: {output_path}")
        
        # Tạo file text summary dễ đọc
        summary_path = output_path.with_suffix('.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("PDF CLASSIFICATION REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Directory: {pdf_dir}\n\n")
            
            f.write("VECTOR PDFs:\n")
            f.write("-"*70 + "\n")
            for pdf in vector_pdfs:
                f.write(f"✅ {pdf['filename']}\n")
                f.write(f"   {pdf['pages']} pages, {pdf['file_size_mb']} MB\n\n")
            
            f.write("\nIMAGE-BASED PDFs:\n")
            f.write("-"*70 + "\n")
            for pdf in image_pdfs:
                f.write(f"🖼️  {pdf['filename']}\n")
                f.write(f"   {pdf['pages']} pages, {pdf['file_size_mb']} MB\n\n")
        
        print(f"✅ Đã lưu summary text vào: {summary_path}")
    
    return {
        'vector_pdfs': vector_pdfs,
        'image_pdfs': image_pdfs,
        'error_pdfs': error_pdfs
    }

if __name__ == "__main__":
    pdf_directory = "/Users/uyenvuong/Documents/CAD/pdf"
    output_file = "/Users/uyenvuong/Documents/CAD/pdf_classification.json"
    
    classify_all_pdfs(pdf_directory, output_file)

