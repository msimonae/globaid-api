# main.py
import os
import re
import io
import requests
import google.generativeai as genai
from urllib.parse import urlparse, quote_plus, parse_qs
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
    description="Uma API para extrair dados de produtos da Amazon, analisar inconsistências e gerar listings otimizados com IA.",
    version="2.6.1", # Versão com extrator de ASIN aprimorado
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

# <<< CORREÇÃO: Função refatorada para ser mais robusta
def extract_product_info_from_url(url: str) -> Optional[dict]:
    """Extrai o ASIN e o país de uma URL da Amazon, com múltiplos métodos de verificação."""
    asin = None
    
    # Método 1: Tenta encontrar padrões como /dp/ASIN ou /gp/product/ASIN
    match = re.search(r"/([dg]p|product)/([A-Z0-9]{10})", url, re.IGNORECASE)
    if match:
        asin = match.group(2)
    else:
        # Método 2: Tenta encontrar um ASIN solto no caminho da URL
        match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url, re.IGNORECASE)
        if match:
            asin = match.group(1)
        else:
            # Método 3 (Fallback): Tenta encontrar o ASIN nos parâmetros da query (ex: ?asin=B09B8MGCTT)
            try:
                parsed_url = urlparse(url)
                query_params = parse_qs(parsed_url.query)
                if 'asin' in query_params and re.match(r'^[A-Z0-9]{10}$', query_params['asin'][0]):
                    asin = query_params['asin'][0]
            except Exception:
                # Se a extração da query falhar, ignora e continua
                pass

    # Se nenhum método encontrou um ASIN, retorna None
    if not asin:
        return None

    # Extrai o país a partir do hostname
    hostname = urlparse(url).hostname
    if not hostname:
        return None

    country_map = {
        "amazon.com.br": "BR", "amazon.com": "US", "amazon.co.uk": "GB",
        "amazon.de": "DE", "amazon.ca": "CA", "amazon.fr": "FR",
        "amazon.es": "ES", "amazon.it": "IT", "amazon.co.jp": "JP",
        "amazon.in": "IN", "amazon.com.mx": "MX", "amazon.com.au": "AU",
    }
    country = next((country_map[key] for key in country_map if key in hostname), "US")
    
    return {"asin": asin, "country": country}


def get_product_details(asin: str, country: str) -> dict:
    # ... (código da função sem alterações)
    return {}

def analyze_product_with_gemini(product_data: dict, country: str) -> str:
    # ... (código da função sem alterações)
    return ""

# ... (Restante do seu código `main.py`, sem alterações) ...
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
