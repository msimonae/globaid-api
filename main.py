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
    version="2.4.0", # Versão com prompt aprimorado
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

# ... (outros modelos Pydantic e funções auxiliares como extract_product_info_from_url, get_product_details, etc. sem alterações) ...

# --- Agente 3: Analisador de Inconsistências (PROMPT ATUALIZADO) ---
def analyze_product_with_gemini(product_data: dict, country: str) -> str:
    if not product_data:
        return "Não foi possível obter os dados do produto para análise."
    # ... (extração de dados como title, features, etc. sem alterações) ...
    title = product_data.get("product_title", "N/A")
    features = "\n- ".join(product_data.get("about_product", []))
    image_urls = product_data.get("product_photos", [])
    if not image_urls:
        return "Produto sem imagens para análise."
        
    # <<< CORREÇÃO: Prompt aprimorado para citar o número da imagem
    prompt_parts = [
        "Você é um analista de controle de qualidade de e-commerce extremamente meticuloso e detalhista.",
        "Sua principal tarefa é encontrar inconsistências factuais entre a descrição textual de um produto e as imagens numeradas que serão fornecidas (Imagem 1, Imagem 2, etc.).",
        "Ao descrever uma discrepância, você DEVE OBRIGATORIAMENTE citar o número da imagem correspondente. Exemplo: 'Na Imagem 2, a altura do produto é de 84,5 cm, o que contradiz o texto.'",
        "Se tudo estiver consistente, declare: 'Nenhuma inconsistência encontrada.'",
        "\n--- DADOS TEXTUAIS DO PRODUTO ---",
        f"**Título:** {title}",
        f"**Destaques (Sobre este item):**\n- {features}",
        "\n--- IMAGENS PARA ANÁLISE VISUAL (numeradas sequencialmente a partir de 1) ---",
    ]
    
    image_count = 0
    for url in image_urls[:5]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            prompt_parts.append(f"--- Imagem {image_count + 1} ---") # Adiciona um rótulo numérico para a IA
            prompt_parts.append(img)
            image_count += 1
        except Exception as e:
            print(f"Aviso: Falha ao processar a imagem {url}. Erro: {e}")
            
    if image_count == 0:
        return "Nenhuma imagem pôde ser baixada para análise."
        
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API do Gemini para análise: {e}")

# --- Endpoints da API ---
@app.post("/analyze", response_model=AnalyzeResponse)
def run_analysis_pipeline(request: AnalyzeRequest):
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

# ... (Restante do código da API FastAPI sem alterações) ...
