# main.py
import os
import re
import io
import requests
import google.generativeai as genai
from urllib.parse import urlparse, quote_plus
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from typing import Optional, List
from PIL import Image

# Carrega as variáveis de ambiente
load_dotenv()

# --- Configuração das Chaves de API ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not RAPIDAPI_KEY or not GEMINI_API_KEY:
    raise RuntimeError("🚨 ALERTA: Chaves de API não encontradas.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"🚨 ALERTA: Falha ao configurar a API do Gemini. Erro: {e}")

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Analisador e Otimizador de Produtos Amazon com IA",
    description="Uma API para extrair dados, analisar inconsistências e otimizar listings.",
    version="2.6.0", # Versão com prompt de análise numérica
)

# --- Modelos Pydantic ---
class AnalyzeRequest(BaseModel):
    amazon_url: HttpUrl

class AnalyzeResponse(BaseModel):
    report: str
    asin: str
    country: str
    product_title: Optional[str] = None
    product_image_url: Optional[str] = None
    product_photos: Optional[List[str]] = []

# ... (outros modelos Pydantic não precisam de alteração) ...

# --- Agentes de Extração de Dados ---
# (As funções extract_product_info_from_url, get_product_details, etc. não precisam de alterações)
def extract_product_info_from_url(url: str) -> Optional[dict]:
    # ... (código sem alterações)
    return {}

def get_product_details(asin: str, country: str) -> dict:
    # ... (código sem alterações)
    return {}


# --- Agente 3: Analisador de Inconsistências (PROMPT REATORADO) ---
def analyze_product_with_gemini(product_data: dict, country: str) -> str:
    if not product_data:
        return "Não foi possível obter os dados do produto para análise."

    # <<< CORREÇÃO: Lógica para extrair as dimensões do texto foi restaurada
    product_dimensions_text = "N/A"
    if info_table := product_data.get("product_information"):
        for key, value in info_table.items():
            if "dimens" in key.lower():
                product_dimensions_text = value
                break

    title = product_data.get("product_title", "N/A")
    features = "\n- ".join(product_data.get("about_product", []))
    image_urls = product_data.get("product_photos", [])
    
    if not image_urls:
        return "Produto sem imagens para análise."

    # <<< CORREÇÃO: Prompt refatorado para ser mais direto e focado em números
    prompt_parts = [
        "Você é um analista de QA de e-commerce extremamente meticuloso e com foco em dados numéricos.",
        "Sua tarefa é comparar os DADOS TEXTUAIS de um produto com as IMAGENS NUMERADAS para encontrar contradições factuais, especialmente em dimensões.",
        "Siga estes passos:",
        "1. Primeiro, analise CADA imagem e extraia todas as especificações numéricas visíveis (altura, largura, profundidade, peso, etc.).",
        "2. Segundo, compare os números extraídos das imagens com os dados fornecidos na seção 'DADOS TEXTUAIS'.",
        "3. Terceiro, se encontrar uma contradição numérica, descreva-a de forma clara e objetiva, mencionando os valores exatos do texto e da imagem.",
        "4. É OBRIGATÓRIO citar o número da imagem onde a inconsistência foi encontrada (ex: 'Na Imagem 2...').",
        "Se tudo estiver consistente, declare: 'Nenhuma inconsistência factual encontrada.'",
        
        "\n--- DADOS TEXTUAIS DO PRODUTO ---",
        f"**Título:** {title}",
        f"**Destaques:**\n- {features}",
        f"**Dimensões do Produto (texto):** {product_dimensions_text}", # Linha crucial que foi restaurada
        
        "\n--- IMAGENS PARA ANÁLISE VISUAL (numeradas sequencialmente a partir de 1) ---",
    ]
    
    image_count = 0
    for url in image_urls[:5]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            prompt_parts.append(f"--- Imagem {image_count + 1} ---")
            prompt_parts.append(img)
            image_count += 1
        except Exception as e:
            print(f"Aviso: Falha ao processar a imagem {url}. Erro: {e}")
            
    if image_count == 0:
        return "Nenhuma imagem pôde ser baixada para análise."
        
    try:
        # Recomendo usar o modelo Pro para tarefas que exigem extração e comparação numérica precisa
        model = genai.GenerativeModel("gemini-1.5-pro-latest") 
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API do Gemini para análise: {e}")

# ... (Restante do código, incluindo get_product_reviews, get_competitors, optimize_listing_with_gemini e os endpoints /analyze e /optimize não precisam de alterações) ...
@app.post("/analyze", response_model=AnalyzeResponse)
def run_analysis_pipeline(request: AnalyzeRequest):
    # ... (código sem alterações)
    url_info = extract_product_info_from_url(str(request.amazon_url))
    if not url_info:
        raise HTTPException(status_code=400, detail="URL inválida ou ASIN não encontrado.")

    product_data = get_product_details(url_info["asin"], url_info["country"])
    analysis_report = analyze_product_with_gemini(product_data, url_info["country"])

    return AnalyzeResponse(
        report=analysis_report,
        asin=url_info["asin"],
        country=url_info["country"],
        product_title=product_data.get("product_title"),
        product_image_url=product_data.get("product_main_image_url"),
        product_photos=product_data.get("product_photos", [])
    )
