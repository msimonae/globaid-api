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
    version="2.3.0",
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

class OptimizeRequest(BaseModel):
    amazon_url: HttpUrl

class OptimizeResponse(BaseModel):
    optimized_listing_report: str
    asin: str
    country: str

# --- Mapeamento de Mercado ---
MARKET_MAP = {
    "BR": ("Português (Brasil)", "Amazon BR"),
    "US": ("English (US)", "Amazon US"),
    "MX": ("Español (México)", "Amazon MX"),
    "ES": ("Español (España)", "Amazon ES"),
}

# --- Agentes de Extração de Dados ---
def extract_product_info_from_url(url: str) -> Optional[dict]:
    asin_match = re.search(r"/([dg]p|product)/([A-Z0-9]{10})", url, re.IGNORECASE)
    if not asin_match:
        asin_match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url, re.IGNORECASE)
        if not asin_match: return None
        asin = asin_match.group(1)
    else:
        asin = asin_match.group(2)
    hostname = urlparse(url).hostname
    if not hostname: return None
    country_map = { "amazon.com.br": "BR", "amazon.com": "US", "amazon.co.uk": "GB", "amazon.de": "DE", "amazon.ca": "CA", "amazon.fr": "FR", "amazon.es": "ES", "amazon.it": "IT", "amazon.co.jp": "JP", "amazon.in": "IN", "amazon.com.mx": "MX", "amazon.com.au": "AU" }
    country = next((country_map[key] for key in country_map if key in hostname), "US")
    return {"asin": asin, "country": country}

def get_product_details(asin: str, country: str) -> dict:
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
    querystring = {"asin": asin, "country": country}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = requests.get(api_url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        data = response.json().get("data")
        if not data: raise HTTPException(status_code=404, detail="Produto não encontrado na API da Amazon.")
        return data
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Erro ao chamar a API da Amazon para detalhes: {e}")

def get_product_reviews(asin: str, country: str) -> dict:
    # ... (código da função sem alterações)
    return {}

def get_competitors(keyword: str, country: str, original_asin: str) -> list:
    # ... (código da função sem alterações)
    return []

# --- Agente 3: Analisador de Inconsistências (FUNÇÃO ATUALIZADA COM SEU PROMPT) ---
def analyze_product_with_gemini(product_data: dict, country: str) -> str: # Adicionado 'country' para manter a assinatura
    if not product_data:
        return "Não foi possível obter os dados do produto para análise."
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
    prompt_parts = [
        "Você é um analista de controle de qualidade de e-commerce extremamente meticuloso e detalhista.",
        "Sua principal tarefa é encontrar inconsistências factuais entre a descrição textual de um produto e suas imagens.",
        "Analise o texto e as imagens a seguir. Aponte QUALQUER discrepância, por menor que seja, entre o que está escrito e o que é mostrado. Preste atenção especial aos dados estruturados, como dimensões, peso, voltagem, cor, material e quantidade de itens.",
        "Se tudo estiver consistente, declare: 'Nenhuma inconsistência encontrada.'",
        "\n--- DADOS TEXTUAIS DO PRODUTO ---",
        f"**Título:** {title}",
        f"**Destaques (Sobre este item):**\n- {features}",
        f"**Dimensões do Produto (extraído da tabela de especificações):** {product_dimensions_text}",
        "\n--- IMAGENS PARA ANÁLISE VISUAL ---",
    ]
    image_count = 0
    for url in image_urls[:5]: # Limita a análise a 5 imagens para performance
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            prompt_parts.append(img)
            image_count += 1
        except requests.exceptions.RequestException as e:
            print(f"Aviso: Falha ao baixar a imagem {url}. Erro: {e}")
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


# --- Agente 4: Otimizador de Listing com Gemini ---
def optimize_listing_with_gemini(product_data: dict, reviews_data: dict, competitors_data: list, url_info: dict) -> str:
    # ... (código da função sem alterações)
    return "Relatório de otimização."


# --- Endpoints da API ---
@app.post("/analyze", response_model=AnalyzeResponse)
def run_analysis_pipeline(request: AnalyzeRequest):
    url_info = extract_product_info_from_url(str(request.amazon_url))
    if not url_info:
        raise HTTPException(status_code=400, detail="URL inválida ou ASIN não encontrado.")

    product_data = get_product_details(url_info["asin"], url_info["country"])
    # Passa o 'country' para a função, mesmo que o prompt novo não o utilize diretamente, para manter a consistência
    analysis_report = analyze_product_with_gemini(product_data, url_info["country"])

    return AnalyzeResponse(
        report=analysis_report,
        asin=url_info["asin"],
        country=url_info["country"],
        product_title=product_data.get("product_title"),
        product_image_url=product_data.get("product_main_image_url"),
        product_photos=product_data.get("product_photos", [])
    )

@app.post("/optimize", response_model=OptimizeResponse)
def run_optimization_pipeline(request: OptimizeRequest):
    # ... (código do endpoint sem alterações)
    return OptimizeResponse(optimized_listing_report="...", asin="...", country="...")
