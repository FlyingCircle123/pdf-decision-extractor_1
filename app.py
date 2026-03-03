# =========================
# IMPORTS
# =========================
import streamlit as st
import PyPDF2
from openai import OpenAI
import os
import time
import json
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="PDF Decision Extractor", layout="wide")
CHUNK_SIZE = 3000  # characters, approximate
MODEL = "gpt-3.5-turbo"

# =========================
# PDF PROCESSING LAYER
# =========================
def extract_text_from_pdf(pdf_file):
    """Extract raw text from uploaded PDF safely."""
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:  # Guard against None returns
            text += page_text
    return text

def extract_text_with_ocr(pdf_file):
    """Fallback: convert PDF pages to images and extract text via OCR with progress."""
    text = ""
    pdf_file.seek(0)  # reset pointer
    images = convert_from_bytes(pdf_file.read())
    
    total_pages = len(images)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, page in enumerate(images):
        status_text.text(f"OCR page {i+1}/{total_pages}...")
        page_text = pytesseract.image_to_string(page)
        if page_text:
            text += page_text + "\n"
        progress_bar.progress((i + 1) / total_pages)
    
    status_text.text("OCR complete!")
    return text

def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split long text into manageable chunks (character-based for MVP)."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# =========================
# AI ENGINE LAYER
# =========================
def build_prompt(chunk):
    """Create structured JSON prompt for consistent extraction."""
    return f"""
    Extract all key decisions, action items, and important points from this text.
    
    Return ONLY valid JSON with this exact structure:
    {{
        "decisions": ["specific decision 1", "specific decision 2"],
        "action_items": ["action item 1", "action item 2"],
        "key_points": ["key point 1", "key point 2"]
    }}
    
    Rules:
    - If a category has no items, use empty list []
    - Be specific and concise
    - Extract directly from the text, don't invent
    - Return ONLY the JSON, no other text
    
    Text: {chunk}
    """

def call_ai(prompt, client):
    """Send prompt to model and return parsed JSON or fallback text."""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        
        # Try to parse as JSON
        try:
            # Clean potential markdown code blocks
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            elif content.startswith("```"):
                content = content.replace("```", "")
            
            return json.loads(content)
        except:
            # Fallback: return as raw text with structure note
            return {"raw": content, "note": "AI didn't return valid JSON"}
            
    except Exception as e:
        return {"error": str(e)}

def merge_results(results):
    """Combine multiple chunk results into one unified structure."""
    merged = {
        "decisions": [],
        "action_items": [],
        "key_points": []
    }
    
    for result in results:
        if isinstance(result, dict):
            # If it's our expected structure
            for key in merged.keys():
                if key in result and isinstance(result[key], list):
                    merged[key].extend(result[key])
            # Handle raw fallback
            if "raw" in result:
                merged["key_points"].append(result["raw"])
    
    # Remove duplicates while preserving order
    for key in merged.keys():
        seen = set()
        unique = []
        for item in merged[key]:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        merged[key] = unique
    
    return merged

def process_document(chunks, client):
    """Process all chunks and merge results."""
    chunk_results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks):
        status_text.text(f"Processing chunk {i+1}/{len(chunks)}...")
        prompt = build_prompt(chunk)
        result = call_ai(prompt, client)
        chunk_results.append(result)
        progress_bar.progress((i + 1) / len(chunks))
    
    status_text.text("Merging results...")
    final_result = merge_results(chunk_results)
    return final_result

# =========================
# UI LAYER
# =========================
def render_output(result):
    """Display structured decision results cleanly."""
    st.markdown("## 📋 Extracted Decisions")
    
    # Handle error case
    if "error" in result:
        st.error(f"Error: {result['error']}")
        return
    
    # Display in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Decisions")
        if result.get("decisions"):
            for d in result["decisions"]:
                st.markdown(f"- {d}")
        else:
            st.markdown("*No decisions found*")
        
        st.markdown("### ⚡ Action Items")
        if result.get("action_items"):
            for a in result["action_items"]:
                st.markdown(f"- {a}")
        else:
            st.markdown("*No action items found*")
    
    with col2:
        st.markdown("### 💡 Key Points")
        if result.get("key_points"):
            for k in result["key_points"]:
                st.markdown(f"- {k}")
        else:
            st.markdown("*No key points found*")
    
    # Prepare text version for download
    text_output = ""
    for category, items in result.items():
        if isinstance(items, list):
            text_output += f"\n{category.upper()}\n"
            text_output += "-" * 20 + "\n"
            for item in items:
                text_output += f"• {item}\n"
            text_output += "\n"
    
    # Download button
    st.download_button(
        label="📥 Download as Text",
        data=text_output,
        file_name="extracted_decisions.txt",
        mime="text/plain"
    )

def main():
    """Main app entry point."""
    st.title("📄 PDF Decision Extractor")
    st.markdown("Upload a PDF to extract key decisions, actions, and insights.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        api_key = st.secrets["OPENAI_API_KEY"]
        st.markdown("---")
        st.markdown("**How it works:**")
        st.markdown("1. Upload PDF")
        st.markdown("2. Text is extracted and chunked")
        st.markdown("3. AI extracts structured data")
        st.markdown("4. Results are merged and displayed")
    
    # Main area
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")
    
    if uploaded_file and api_key:
        if st.button("🚀 Extract Decisions", type="primary"):
            with st.spinner("Processing your PDF..."):
                # Initialize client
                client = OpenAI(api_key=api_key)
                
                # Extract text
                text = extract_text_from_pdf(uploaded_file)
                
                if not text.strip():
                    st.warning("No text detected — trying OCR fallback...")
                    text = extract_text_with_ocr(uploaded_file)
                
                if not text.strip():
                    st.error("OCR also failed. Cannot extract text from this PDF.")
                    st.stop()
                
                # Show stats for whichever extraction succeeded
                st.info(f"📊 Extracted {len(text)} characters, {len(text.split())} words")
                
                # Chunk text
                chunks = chunk_text(text)
                st.info(f"📦 Split into {len(chunks)} chunks for processing")
                
                # Process with AI
                result = process_document(chunks, client)
                
                # Display
                render_output(result)
                
                # Success message
                st.success("✅ Extraction complete!")
    
    elif uploaded_file and not api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar.")
    
    elif not uploaded_file:
        st.info("👆 Start by uploading a PDF file")

if __name__ == "__main__":
    main()
