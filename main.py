# main_async.py
import os
import re
import io
import httpx # <<< MUDANÇA: Importa httpx
import asyncio # <<< MUDANÇA: Importa asyncio
from urllib.parse import urlparse, quote_plus, parse_qs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from typing import Optional, List
from PIL import Image
from openai import AsyncOpenAI # <<< MUDANÇA: Cliente Async do OpenAI

# Carrega as variáveis de ambiente
load_dotenv()

# --- Configuração das Chaves de API ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not RAPIDAPI_KEY or not OPENROUTER_API_KEY:
    raise RuntimeError("🚨 ALERTA: Chaves de API não encontradas. Verifique .env")

# <<< MUDANÇA: Configura o cliente Async da API do OpenRouter
try:
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    print("✅ API Async do OpenRouter configurada com sucesso.")
except Exception as e:
    raise RuntimeError(f"🚨 ALERTA: Falha ao configurar a API do OpenRouter. Erro: {e}")

# <<< MUDANÇA: Cria um cliente HTTP assíncrono global para reutilizar conexões
# Isso é uma boa prática de performance.
http_client = httpx.AsyncClient(timeout=45.0) # Timeout de 45s por request

# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Analisador e Otimizador de Produtos Amazon com IA",
    description="Uma API para extrair dados, analisar inconsistências e otimizar listings.",
    version="4.0.0",
)

# --- Modelos Pydantic (sem alterações) ---
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

# --- Mapeamento de Mercado (sem alterações) ---
MARKET_MAP = {
    "BR": ("Português (Brasil)", "Amazon BR"),
    "US": ("English (US)", "Amazon US"),
    "MX": ("Español (México)", "Amazon MX"),
    "ES": ("Español (España)", "Amazon ES"),
}

# --- Agentes de Extração de Dados (Funções que fazem I/O agora são async) ---
def extract_product_info_from_url(url: str) -> Optional[dict]:
    # Esta função não faz I/O, então permanece síncrona
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

# <<< MUDANÇA: Função convertida para async usando httpx
async def get_product_details(asin: str, country: str) -> dict:
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
    querystring = {"asin": asin, "country": country}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = await http_client.get(api_url, headers=headers, params=querystring)
        response.raise_for_status()
        data = response.json().get("data")
        if not data: raise HTTPException(status_code=404, detail="Produto não encontrado na API da Amazon.")
        return data
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Erro ao chamar a API da Amazon para detalhes: {e}")


# <<< MUDANÇA: Função convertida para async (embora não seja usada no /batch_analyze, é boa prática)
async def get_product_reviews(asin: str, country: str) -> dict:
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-reviews"
    querystring = {"asin": asin, "country": country, "sort_by": "recent", "page_size": "20"}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = await http_client.get(api_url, headers=headers, params=querystring)
        response.raise_for_status()
        reviews = response.json().get("data", {}).get("reviews", [])
        positive = [r['review_comment'] for r in reviews if r['review_star_rating'] >= 4]
        negative = [r['review_comment'] for r in reviews if r['review_star_rating'] <= 2]
        return {"positive_reviews": positive[:10], "negative_reviews": negative[:10]}
    except httpx.RequestError:
        return {"positive_reviews": [], "negative_reviews": []}

# <<< MUDANÇA: Função convertida para async
async def get_competitors(keyword: str, country: str, original_asin: str) -> list:
    api_url = "https://real-time-amazon-data.p.rapidapi.com/search"
    querystring = {"query": quote_plus(keyword), "country": country, "page_size":"10"}
    headers = {"x-rapidapi-key": RAPIDAPI_KEY, "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com"}
    try:
        response = await http_client.get(api_url, headers=headers, params=querystring)
        response.raise_for_status()
        products = response.json().get("data", {}).get("products", [])
        competitors = []
        for p in products:
            if p.get('asin') != original_asin and not p.get('is_sponsored', False):
                competitors.append({"title": p.get('product_title'), "price": p.get('product_price'), "rating": p.get('product_star_rating'), "reviews_count": p.get('product_num_ratings')})
            if len(competitors) >= 5: break
        return competitors
    except httpx.RequestError:
        return []

# <<< MUDANÇA: Analisador do Gemini convertido para async
async def analyze_product_with_gemini(product_data: dict, country: str) -> str:
    # ... (lógica interna para montar o prompt é a mesma, sem alterações)
    product_dimensions_text = "N/A"
    info_table = product_data.get("product_information") or {}
    for key, value in info_table.items():
        if "dimens" in key.lower():
            product_dimensions_text = value
            break
    title = product_data.get("product_title", "N/A")
    description = product_data.get("product_description", "")
    features = product_data.get("about_product", []) or []
    features_text = "\n- ".join(features) if features else "N/A"
    full_text_content = f"{description}\n\nFeatures:\n- {features_text}".strip()
    image_urls = product_data.get("product_photos", []) or []
    if not image_urls:
        return (f"⚠️ Nenhuma imagem de produto foi retornada pela API.\n"
                f"--- DADOS TEXTUAIS ---\n"
                f"**Título:** {title}\n"
                f"**Conteúdo do anúncio:**\n{full_text_content}\n"
                f"**Dimensões (texto):** {product_dimensions_text}")
    prompt_parts = [
        "Você é um analista de QA de e-commerce...",
        # ... (todo o seu prompt continua o mesmo) ...
        f"**Dimensões do Produto (texto):** {product_dimensions_text}",
        "\n--- IMAGENS PARA ANÁLISE VISUAL (numeradas sequencialmente a partir de 1) ---",
    ]
    for i, url in enumerate(image_urls[:5], start=1):
        prompt_parts.append(f"Imagem {i}: {url}")
    prompt_text = "\n".join(prompt_parts)

    try:
        # <<< MUDANÇA: Usa 'await' e o cliente async
        response = await client.chat.completions.create(
            model="gemini-2.5-pro", 
            messages=[{"role": "user", "content": prompt_text}],
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API para análise: {e}")

# <<< MUDANÇA: Otimizador do Gemini convertido para async
async def optimize_listing_with_gemini(product_data: dict, reviews_data: dict, competitors_data: list, url_info: dict) -> str:
    lang, market = MARKET_MAP.get(url_info["country"], ("English (US)", f"Amazon {url_info['country']}"))
    user_content = [ f"Você é um Consultor Sênior de E-commerce...", 
    # ... (seu prompt continua o mesmo) ...
    ]
    try:
        # <<< MUDANÇA: Usa 'await' e o cliente async
        response = await client.chat.completions.create(
            model="gemini-2.5-pro", 
            messages=[{"role": "user", "content": "\n".join(user_content)}]
        )
        return response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao chamar a API para otimização: {e}")

# <<< MUDANÇA: Função de processamento refatorada para ser async
async def process_single_url_async(url: str) -> AnalyzeResponse:
    url_info = extract_product_info_from_url(url)
    if not url_info:
        return AnalyzeResponse(report=f"Erro: URL inválida ou ASIN não encontrado.", asin="ERRO", country="N/A", product_title=f"Falha ao processar URL: {url}")
    
    try:
        # <<< MUDANÇA: Usa 'await' nas chamadas de I/O
        product_data = await get_product_details(url_info["asin"], url_info["country"])
        analysis_report = await analyze_product_with_gemini(product_data, url_info["country"])
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
            report=f"Erro ao processar o ASIN {url_info.get('asin', 'N/A')}: {error_detail}",
            asin=url_info.get('asin', 'ERRO'),
            country=url_info.get('country', 'N/A'),
            product_title=f"Falha ao processar URL: {url}"
        )

# --- Endpoints da API ---

# Endpoint original mantido, mas agora é async e chama a função async
@app.post("/analyze", response_model=AnalyzeResponse)
async def run_analysis_pipeline(request: AnalyzeRequest):
    result = await process_single_url_async(str(request.amazon_url))
    if result.asin == "ERRO":
        raise HTTPException(status_code=400, detail=result.report)
    return result

# <<< GRANDE MUDANÇA: Endpoint de lote agora é async e usa asyncio.gather
@app.post("/batch_analyze", response_model=BatchAnalyzeResponse)
async def run_batch_analysis_pipeline(request: BatchAnalyzeRequest):
    # Cria uma lista de "tarefas" (coroutines) para cada URL
    tasks = [process_single_url_async(str(url)) for url in request.amazon_urls]
    
    # Executa todas as tarefas concorrentemente e aguarda a conclusão de todas
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Tratamento de erro caso alguma das tarefas falhe
    processed_results = []
    for result in results:
        if isinstance(result, Exception):
            # Se uma tarefa específica falhou, criamos um objeto de resposta de erro
            processed_results.append(AnalyzeResponse(
                report=f"Erro crítico durante o processamento concorrente: {str(result)}",
                asin="ERRO_FATAL",
                country="N/A"
            ))
        else:
            processed_results.append(result)

    return BatchAnalyzeResponse(results=processed_results)

# Endpoint de otimização também precisa ser async
@app.post("/optimize", response_model=OptimizeResponse)
async def run_optimization_pipeline(request: OptimizeRequest):
    url_info = extract_product_info_from_url(str(request.amazon_url))
    if not url_info:
        raise HTTPException(status_code=400, detail="URL inválida ou ASIN não encontrado.")

    asin, country = url_info["asin"], url_info["country"]

    # Executa as chamadas de API de forma concorrente onde for possível
    product_data, reviews_data = await asyncio.gather(
        get_product_details(asin, country),
        get_product_reviews(asin, country)
    )

    keyword = product_data.get("product_title", asin)
    competitors_data = await get_competitors(keyword, country, asin)

    optimization_report = await optimize_listing_with_gemini(product_data, reviews_data, competitors_data, url_info)

    return OptimizeResponse(
        optimized_listing_report=optimization_report,
        asin=asin,
        country=country
    )

