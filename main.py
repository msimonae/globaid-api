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

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# --- Configuração das Chaves de API de forma segura ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not RAPIDAPI_KEY or not GEMINI_API_KEY:
    raise RuntimeError("🚨 ALERTA: Chaves de API não encontradas. Verifique seu arquivo .env ou as variáveis de ambiente do servidor.")

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"🚨 ALERTA: Falha ao configurar a API do Gemini. Erro: {e}")

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Analisador e Otimizador de Produtos Amazon com IA",
    description="Uma API para extrair dados de produtos da Amazon, analisar inconsistências e gerar listings otimizados com IA.",
    version="2.2.0",
)

# --- Modelos Pydantic para validação ---
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

# --- Mapeamento de Mercado (Constante) ---
MARKET_MAP = {
    "BR": ("Português (Brasil)", "Amazon BR"),
    "US": ("English (US)", "Amazon US"),
    "MX": ("Español (México)", "Amazon MX"),
    "ES": ("Español (España)", "Amazon ES"),
}

# --- Agentes de Extração de Dados ---

def extract_product_info_from_url(url: str) -> Optional[dict]:
    """Extrai o ASIN e o código do país de uma URL da Amazon."""
    asin_match = re.search(r"/([dg]p|product)/([A-Z0-9]{10})", url, re.IGNORECASE)
    if not asin_match:
        asin_match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url, re.IGNORECASE)
        if not asin_match: return None
        asin = asin_match.group(1)
    else:
        asin = asin_match.group(2)
    hostname = urlparse(url).hostname
    if not hostname: return None
    country_map = {
        "amazon.com.br": "BR", "amazon.com": "US", "amazon.co.uk": "GB",
        "amazon.de": "DE", "amazon.ca": "CA", "amazon.fr": "FR",
        "amazon.es": "ES", "amazon.it": "IT", "amazon.co.jp": "JP",
        "amazon.in": "IN", "amazon.com.mx": "MX", "amazon.com.au": "AU",
    }
    country = next((country_map[key] for key in country_map if key in hostname), "US")
    return {"asin": asin, "country": country}

def get_product_details(asin: str, country: str) -> dict:
    """Busca detalhes de um produto na Amazon usando a API da RapidAPI."""
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
    querystring = {"asin": asin, "country": country}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = requests.get(api_url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        data = response.json().get("data")
        if not data:
            raise HTTPException(status_code=404, detail="Produto não encontrado na API da Amazon.")
        return data
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Erro ao chamar a API da Amazon para detalhes: {e}")

def get_product_reviews(asin: str, country: str) -> dict:
    """Busca os reviews de um produto."""
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"
    querystring = {"asin": asin, "country": country, "sort_by": "recent", "page_size": "20"}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = requests.get(api_url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        reviews = response.json().get("data", {}).get("reviews", [])
        positive = [r['review_comment'] for r in reviews if r['review_star_rating'] >= 4]
        negative = [r['review_comment'] for r in reviews if r['review_star_rating'] <= 2]
        return {"positive_reviews": positive[:10], "negative_reviews": negative[:10]}
    except requests.exceptions.RequestException:
        return {"positive_reviews": [], "negative_reviews": []}

def get_competitors(keyword: str, country: str, original_asin: str) -> list:
    """Busca produtos na Amazon por palavra-chave para encontrar concorrentes."""
    api_url = "https://real-time-amazon-data.p.rapidapi.com/search"
    querystring = {"query": quote_plus(keyword), "country": country, "page_size":"10"}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = requests.get(api_url, headers=headers, params=querystring, timeout=30)
        response.raise_for_status()
        products = response.json().get("data", {}).get("products", [])
        competitors = []
        for p in products:
            if p.get('asin') != original_asin and not p.get('is_sponsored', False):
                competitors.append({
                    "title": p.get('product_title'), "price": p.get('product_price'),
                    "rating": p.get('product_star_rating'), "reviews_count": p.get('product_num_ratings')
                })
            if len(competitors) >= 5: break
        return competitors
    except requests.exceptions.RequestException:
        return []

# --- Agentes de IA ---

def analyze_product_with_gemini(product_data: dict, country: str) -> str:
    """Usa a IA do Gemini para analisar inconsistências entre texto e imagens (internacionalizado)."""
    lang, _ = MARKET_MAP.get(country, ("English (US)", f"Amazon {country}"))
    prompt_texts = {
        "Português (Brasil)": {"persona": "Você é um analista de QA de e-commerce meticuloso...", "task": "Sua tarefa é encontrar inconsistências...", "instructions": "Analise o texto e as imagens...", "success_case": "Se consistente, declare: 'Nenhuma inconsistência encontrada.'", "text_header": "DADOS TEXTUAIS", "title_label": "Título", "features_label": "Destaques", "dimensions_label": "Dimensões", "image_header": "IMAGENS"},
        "English (US)": {"persona": "You are a meticulous e-commerce QA analyst...", "task": "Your task is to find inconsistencies...", "instructions": "Analyze the text and images...", "success_case": "If consistent, state: 'No inconsistencies found.'", "text_header": "TEXTUAL DATA", "title_label": "Title", "features_label": "Highlights", "dimensions_label": "Dimensions", "image_header": "IMAGES"},
        "Español (México)": {"persona": "Eres un analista de QA de e-commerce meticuloso...", "task": "Tu tarea es encontrar inconsistencias...", "instructions": "Analiza el texto y las imágenes...", "success_case": "Si es consistente, declara: 'No se encontraron inconsistencias.'", "text_header": "DATOS TEXTUALES", "title_label": "Título", "features_label": "Puntos clave", "dimensions_label": "Dimensiones", "image_header": "IMÁGENES"},
    }
    pt = prompt_texts.get(lang, prompt_texts["English (US)"])
    title = product_data.get("product_title", "N/A")
    features = "\n- ".join(product_data.get("about_product", []))
    image_urls = product_data.get("product_photos", [])
    prompt_parts = [
        pt["persona"], pt["task"], pt["instructions"], pt["success_case"],
        f"\n--- {pt['text_header']} ---", f"**{pt['title_label']}:** {title}",
        f"**{pt['features_label']}:**\n- {features}", f"\n--- {pt['image_header']} ---",
    ]
    for url in image_urls[:5]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            prompt_parts.append(Image.open(io.BytesIO(response.content)))
        except Exception: continue
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API do Gemini para análise: {e}")

def optimize_listing_with_gemini(product_data: dict, reviews_data: dict, competitors_data: list, url_info: dict) -> str:
    """Usa a IA do Gemini com o prompt estratégico REFORÇADO para otimizar o listing."""
    lang, market = MARKET_MAP.get(url_info["country"], ("English (US)", f"Amazon {url_info['country']}"))
    prompt = [
        f"Você é um Consultor Sênior de E-commerce, mestre em SEO para o ecossistema Amazon (A9, Rufus). Sua missão é otimizar um listing para maximizar vendas no mercado {market}.",
        f"A resposta DEVE ser inteiramente em {lang}.",
        f"--- DADOS DO PRODUTO ATUAL ---\nTítulo: {product_data.get('product_title', 'N/A')}\nFeatures: {product_data.get('about_product', [])}",
        f"--- INTELIGÊNCIA DE MERCADO ---\nReviews Positivos: {reviews_data.get('positive_reviews')}\nReviews Negativos: {reviews_data.get('negative_reviews')}\nConcorrentes: {competitors_data}",
        "\n--- INSTRUÇÕES E FORMATO DE SAÍDA OBRIGATÓRIO ---",
        "Gere sua resposta seguindo ESTRITAMENTE a estrutura Markdown abaixo, sem omitir nenhuma seção. Use os títulos exatamente como especificados.",
        "### 1. Título Otimizado (SEO)\n[Gere aqui o título otimizado]",
        "### 2. Feature Bullets Otimizados (5 Pontos)\n[Gere aqui os 5 feature bullets, um por linha]",
        "### 3. Descrição do Produto (Estrutura para A+ Content)\n[Gere aqui a descrição persuasiva]",
        "### 4. Análise Competitiva e Estratégia\n[Gere aqui a tabela comparativa e o parágrafo de estratégia]",
        "### 5. Sugestões de Palavras-chave (Backend)\n[Gere aqui a lista de 15-20 palavras-chave long-tail]",
        "### 6. FAQ Estratégico (Top 5 Perguntas e Respostas)\n[Gere aqui as 5 Q&As]",
        "\n--- REGRAS INQUEBRÁVEIS ---\n- Não invente características. Use apenas os dados fornecidos.\n- Não use clichês genéricos. Seja específico e factual.\n- O conteúdo final deve ser único e superior ao dos concorrentes."
    ]
    try:
        model = genai.GenerativeModel("gemini-1.5-pro-latest")
        response = model.generate_content("\n".join(prompt))
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API do Gemini para otimização: {e}")

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

@app.post("/optimize", response_model=OptimizeResponse)
def run_optimization_pipeline(request: OptimizeRequest):
    url_info = extract_product_info_from_url(str(request.amazon_url))
    if not url_info:
        raise HTTPException(status_code=400, detail="URL inválida ou ASIN não encontrado.")
    
    asin, country = url_info["asin"], url_info["country"]
    
    product_data = get_product_details(asin, country)
    reviews_data = get_product_reviews(asin, country)
    keyword = product_data.get("product_title", asin)
    competitors_data = get_competitors(keyword, country, asin)
    
    optimization_report = optimize_listing_with_gemini(product_data, reviews_data, competitors_data, url_info)

    return OptimizeResponse(
        optimized_listing_report=optimization_report,
        asin=asin,
        country=country
    )
