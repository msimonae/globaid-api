# main.py
import os
import re
import io
import requests
from urllib.parse import urlparse, quote_plus, parse_qs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from typing import Optional, List
from PIL import Image
from openai import OpenAI

# Carrega as variáveis de ambiente
load_dotenv()

# --- Configuração das Chaves de API ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not RAPIDAPI_KEY or not OPENROUTER_API_KEY:
    raise RuntimeError("🚨 ALERTA: Chaves de API não encontradas. Verifique .env")

# Configura o cliente da API do OpenRouter
try:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    print("✅ API do OpenRouter configurada com sucesso.")
except Exception as e:
    raise RuntimeError(f"🚨 ALERTA: Falha ao configurar a API do OpenRouter. Erro: {e}")


# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Analisador e Otimizador de Produtos Amazon com IA",
    description="Uma API para extrair dados, analisar inconsistências e otimizar listings.",
    version="4.0.0",
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
    product_features: Optional[List[str]] = []

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
    asin = None
    match = re.search(r"/([dg]p|product)/([A-Z0-9]{10})", url, re.IGNORECASE)
    if match:
        asin = match.group(2)
    else:
        match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url, re.IGNORECASE)
        if match:
            asin = match.group(1)
        else:
            try:
                parsed_url = urlparse(url)
                query_params = parse_qs(parsed_url.query)
                if 'asin' in query_params and re.match(r'^[A-Z0-9]{10}$', query_params['asin'][0]):
                    asin = query_params['asin'][0]
            except Exception:
                pass
    if not asin:
        return None
    hostname = urlparse(url).hostname
    if not hostname:
        return None
    country_map = {"amazon.com.br": "BR", "amazon.com": "US", "amazon.co.uk": "GB", "amazon.de": "DE", "amazon.ca": "CA", "amazon.fr": "FR", "amazon.es": "ES", "amazon.it": "IT", "amazon.co.jp": "JP", "amazon.in": "IN", "amazon.com.mx": "MX", "amazon.com.au": "AU"}
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

# --- Agente 3: Analisador de Inconsistências (FUNÇÃO ATUALIZADA) ---
def analyze_product_with_gemini(product_data: dict, country: str) -> str:
    """
    Versão corrigida: não envia objetos PIL.Image para a API.
    Em vez disso envia um prompt textual que inclui os URLs das imagens numeradas.
    """
    product_dimensions_text = "N/A"
    info_table = product_data.get("product_information") or {}
    for key, value in info_table.items():
        if "dimens" in key.lower():
            product_dimensions_text = value
            break

    # Dados textuais do produto
    # Dados textuais principais
    title = product_data.get("product_title", "N/A")
    description = product_data.get("product_description", "")
    features = product_data.get("about_product", []) or []
    features_text = "\n- ".join(features) if features else "N/A"

    # Monta conteúdo textual consolidado
    full_text_content = f"{description}\n\nFeatures:\n- {features_text}".strip()

    # Lista de imagens
    image_urls = product_data.get("product_photos", []) or []

    # Se não houver imagens, ainda assim retorna relatório textual
    if not image_urls:
        return (
            f"⚠️ Nenhuma imagem de produto foi retornada pela API.\n"
            f"--- DADOS TEXTUAIS ---\n"
            f"**Título:** {title}\n"
            f"**Conteúdo do anúncio:**\n{full_text_content}\n"
            f"**Dimensões (texto):** {product_dimensions_text}"
        )

    prompt_parts = [
        "Você é um analista de QA de e-commerce extremamente meticuloso e com foco em dados numéricos.",
        "Priorize a busca por inconsistências em especificações técnicas, recursos, nomes e funcionalidades. Além disso, verifique se existem informações que aparentam ser equivocadas ou erradas a respeito dos produtos.",
        "Sua tarefa é comparar os DADOS TEXTUAIS de um produto com as IMAGENS NUMERADAS para encontrar contradições factuais, especialmente em dimensões, dados específicos dos produtos.",
        "Siga estes passos:",
        "1. Primeiro, analise CADA imagem e extraia todas as especificações numéricas visíveis (altura, largura, profundidade, peso, etc.).",
        "2. Segundo, compare os números extraídos das imagens com os dados fornecidos na seção 'DADOS TEXTUAIS'.",
        "3. Terceiro, se encontrar uma contradição numérica, descreva-a de forma clara e objetiva, mencionando os valores exatos do texto e da imagem.",
        "4. É OBRIGATÓRIO citar o número da imagem onde a inconsistência foi encontrada (ex: 'Na Imagem 2...').",
        "5. Analise e compare os Dados do Listing - Conteúdo textual do anúncio e Dimensões do Produto (texto). Crie um relatório claro e conciso listando TODAS as discrepâncias encontradas.",
        "Discrepâncias podem ser:\n"
        "- Informações contraditórias (ex: texto diz 'bateria de 10h', imagem mostra 'bateria de 8h').\n"
        "- Recursos mencionados no texto mas não mostrados ou validados nas imagens.\n"
        "- Recursos ou textos importantes visíveis nas imagens mas não mencionados na descrição textual.\n"
        "- Preste muita atenção a detalhes técnicos, como dimensões, peso, material, etc, nas imagens que estejam possivelmente inconsistentes com as informações textuais.\n"
        "- Qualquer erro ou inconsistência que possa afetar a decisão de compra do cliente.\n"
        "- Se houver discrepâncias, forneça uma explicação clara do porquê de cada uma ser considerada uma discrepância.\n"
        "- Agrupe as discrepâncias por tipo, se possível, para facilitar a análise.",
        "Se tudo estiver consistente, declare: 'Nenhuma inconsistência factual encontrada.'",
        "\n--- DADOS TEXTUAIS DO PRODUTO ---",
        f"**Título:** {title}",
        f"**Dados do Listing - Conteúdo textual do anúncio:**\n{full_text_content}",
        f"**Dimensões do Produto (texto):** {product_dimensions_text}",
        "\n--- IMAGENS PARA ANÁLISE VISUAL (numeradas sequencialmente a partir de 1) ---",
    ]

    for i, url in enumerate(image_urls[:5], start=1):
        prompt_parts.append(f"Imagem {i}: {url}")

    # Junta tudo em uma string final
    prompt_text = "\n".join(prompt_parts)

    try:
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[{"role": "user", "content": prompt_text}],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API para análise: {e}")

# --- Agente 4: Otimizador de Listing com Gemini (FUNÇÃO ATUALIZADA) ---
def optimize_listing_with_gemini(product_data: dict, reviews_data: dict, competitors_data: list, url_info: dict) -> str:
    lang, market = MARKET_MAP.get(url_info["country"], ("English (US)", f"Amazon {url_info['country']}"))
    
    user_content = [
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
        response = client.chat.completions.create(
            model="google/gemini-2.5-pro",
            messages=[
                {"role": "user", "content": user_content}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API para otimização: {e}")

# <<< NOVO: Função refatorada para processar uma única URL (evita duplicação de código)
def process_single_url(url: str) -> AnalyzeResponse:
    """Contém a lógica completa de análise para uma única URL."""
    url_info = extract_product_info_from_url(url)
    if not url_info:
        return AnalyzeResponse(
            report=f"Erro: URL inválida ou ASIN não encontrado na URL fornecida.",
            asin="ERRO", country="N/A", product_title=f"Falha ao processar URL: {url}",
            product_photos=[], product_features=[]
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
        error_detail = getattr(e, 'detail', str(e))
        return AnalyzeResponse(
            report=f"Erro ao processar o produto com ASIN {url_info.get('asin', 'N/A')}: {error_detail}",
            asin=url_info.get('asin', 'ERRO'),
            country=url_info.get('country', 'N/A'),
            product_title=f"Falha ao processar URL: {url}",
            product_photos=[], product_features=[]
        )
    
# --- Endpoints da API ---
@app.post("/analyze", response_model=AnalyzeResponse)
def run_analysis_pipeline(request: AnalyzeRequest):
    result = process_single_url(str(request.amazon_url))
    if result.asin == "ERRO":
        raise HTTPException(status_code=400, detail=result.report)
    return result
    
# <<< NOVO: Endpoint para processar uma lista de URLs em lote
@app.post("/batch_analyze", response_model=BatchAnalyzeResponse)
def run_batch_analysis_pipeline(request: BatchAnalyzeRequest):
    """Endpoint para analisar uma lista de URLs em lote, agora com tratamento de erro."""
    try:
        # Processa cada URL individualmente e coleta os resultados
        results = [process_single_url(str(url)) for url in request.amazon_urls]
        return BatchAnalyzeResponse(results=results)
    except Exception as e:
        # Se ocorrer um erro inesperado durante o processamento em lote,
        # retorna um erro 500 com uma mensagem clara.
        raise HTTPException(
            status_code=500,
            detail=f"Ocorreu um erro interno inesperado durante o processamento em lote: {str(e)}"
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






