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
    version="2.0.0",
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
    # <<< NOVO EM VERSÃO ANTERIOR: Adicionado para o PDF
    product_features: Optional[List[str]] = []

# <<< NOVO: Modelos para a funcionalidade de lote
class BatchAnalyzeRequest(BaseModel):
    amazon_urls: List[HttpUrl]

class BatchAnalyzeResponse(BaseModel):
    results: List[AnalyzeResponse]

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
                competitors.append({"title": p.get('product_title'), "price": p.get('product_price'), "rating": p.get('product_star_rating'), "reviews_count": p.get('product_num_ratings')})
            if len(competitors) >= 5: break
        return competitors
    except requests.exceptions.RequestException:
        return []
# --- Agente 3: Analisador de Inconsistências (FUNÇÃO ATUALIZADA COM SEU PROMPT) ---
def analyze_product_with_gemini(product_data: dict, country: str) -> str:
    if not product_data:
        return "Não foi possível obter os dados do produto para análise."

    # Extração de dados (incluindo os novos campos)
    title = product_data.get("product_title", "N/A")
    features = "\n- ".join(product_data.get("about_product", []))
    image_urls = product_data.get("product_photos", [])
    
    # <<< NOVO: Extração dos campos adicionais
    product_description = product_data.get("product_description", "Nenhuma descrição longa fornecida.")
    
    # Formata os dicionários 'product_information' e 'product_details' para texto legível
    product_information = product_data.get("product_information", {})
    info_formatted = "\n".join([f"- {key}: {value}" for key, value in product_information.items()]) if product_information else "N/A"

    product_details = product_data.get("product_details", {})
    details_formatted = "\n".join([f"- {key}: {value}" for key, value in product_details.items()]) if product_details else "N/A"

    # A extração de 'product_dimensions_text' já estava sendo feita a partir de 'product_information'
    product_dimensions_text = "N/A"
    if product_information:
        for key, value in product_information.items():
            if "dimens" in key.lower():
                product_dimensions_text = value
                break

    if not image_urls:
        return "Produto sem imagens para análise."

    # <<< CORREÇÃO: Prompt atualizado para incluir os novos campos
    prompt_parts = [
        "Você é um analista de controle de qualidade de e-commerce extremamente meticuloso e detalhista.",
        "Sua principal tarefa é encontrar inconsistências factuais entre a descrição textual de um produto e suas imagens.",
        "Analise o texto e as imagens a seguir. Aponte QUALQUER discrepância, por menor que seja, entre o que está escrito e o que é mostrado. Preste atenção especial aos dados estruturados, como dimensões, peso, voltagem, cor, material e quantidade de itens, comparando todas as fontes de texto com as imagens.",
        "Se tudo estiver consistente, declare: 'Nenhuma inconsistência encontrada.'",
        
        "\n--- DADOS TEXTUAIS DO PRODUTO ---",
        f"**Título:** {title}",
        f"**Destaques (Sobre este item):**\n- {features}",
        f"**Descrição Longa do Produto:**\n{product_description}",
        f"**Dimensões (extraído da tabela):** {product_dimensions_text}",
        f"**Tabela 'Informação do produto':**\n{info_formatted}",
        f"**Tabela 'Detalhes do produto':**\n{details_formatted}",
        
        "\n--- IMAGENS PARA ANÁLISE VISUAL ---",
    ]
    
    image_count = 0
    for url in image_urls[:5]:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
            prompt_parts.append(img)
            image_count += 1
 #       except requests.exceptions.RequestException as e:
 #           print(f"Aviso: Falha ao baixar a imagem {url}. Erro: {e}")
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

# <<< NOVO: Função refatorada para processar uma única URL (evita duplicação de código)
def process_single_url(url: str) -> AnalyzeResponse:
    """Contém a lógica completa de análise para uma única URL."""
    url_info = extract_product_info_from_url(url)
    if not url_info:
        # Em um cenário de lote, em vez de lançar exceção, retornamos um relatório de erro
        return AnalyzeResponse(
            report=f"Erro: URL inválida ou ASIN não encontrado na URL fornecida.",
            asin="ERRO", country="N/A", product_title=f"Falha ao processar URL: {url}"
        )

    try:
        product_data = get_product_details(url_info["asin"], url_info["country"])
        analysis_report = analyze_product_with_gemini(product_data, url_info["country"])

        return AnalyzeResponse(
            report=analysis_report,
            asin=url_info["asin"],
            country=url_info["country"],
            product_title=product_data.get("product_title"),
            product_image_url=product_data.get("product_main_image_url"),
            product_photos=product_data.get("product_photos", []),
            product_features=product_data.get("about_product", [])
        )
    except Exception as e:
        return AnalyzeResponse(
            report=f"Erro ao processar o produto com ASIN {url_info.get('asin', 'N/A')}: {str(e)}",
            asin=url_info.get('asin', 'ERRO'),
            country=url_info.get('country', 'N/A'),
            product_title=f"Falha ao processar URL: {url}"
        )


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
        product_photos=product_data.get("product_photos", []),
        product_features=product_data.get("about_product", [])
    )

# <<< NOVO: Endpoint para processar uma lista de URLs em lote
@app.post("/batch_analyze", response_model=BatchAnalyzeResponse)
def run_batch_analysis_pipeline(request: BatchAnalyzeRequest):
    """Endpoint para analisar uma lista de URLs em lote."""
    results = []
    for url in request.amazon_urls:
        # Processa cada URL individualmente e adiciona o resultado à lista
        result = process_single_url(str(url))
        results.append(result)
    return BatchAnalyzeResponse(results=results)


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

