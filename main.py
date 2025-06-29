import os
import re
import requests
import io
import google.generativeai as genai
from urllib.parse import urlparse
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv
from typing import Optional, List
from PIL import Image  # <--- Adicionado para manipulação de imagens

# Carrega as variáveis de ambiente do arquivo .env (para desenvolvimento local)
load_dotenv()

# --- Configuração das Chaves de API de forma segura ---
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not RAPIDAPI_KEY or not GEMINI_API_KEY:
    raise RuntimeError(  # Lança um erro para parar a execução se as chaves não existirem
        "🚨 ALERTA: Chaves de API não encontradas. Verifique seu arquivo .env ou as variáveis de ambiente do servidor."
    )

# --- CORREÇÃO 1: Configuração explícita e recomendada da API do Gemini ---
try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    raise RuntimeError(f"🚨 ALERTA: Falha ao configurar a API do Gemini. Erro: {e}")


# Inicializa a aplicação FastAPI
app = FastAPI(
    title="Analisador de Produtos Amazon com IA",
    description="Uma API para extrair dados de produtos da Amazon e gerar relatórios de inconsistência com Gemini, alinhada com a lógica do Colab.",
    version="1.2.0",  # Versão incrementada com as correções
)


# --- Modelos Pydantic para validação de entrada e saída ---
class AnalyzeRequest(BaseModel):
    amazon_url: HttpUrl


class AnalyzeResponse(BaseModel):
    report: str
    asin: str
    country: str
    product_image_url: Optional[str] = None
    product_photos: Optional[List[str]] = None


# --- Agente 1: Extrator de Informações da URL (sem alterações) ---
def extract_product_info_from_url(url: str) -> Optional[dict]:
    """Extrai o ASIN e o código do país de uma URL da Amazon."""
    # Sua lógica original está correta
    asin_match = re.search(r"/([dg]p|product)/([A-Z0-9]{10})", url, re.IGNORECASE)
    if not asin_match:
        asin_match = re.search(r"/([A-Z0-9]{10})(?:[/?]|$)", url, re.IGNORECASE)
        if not asin_match:
            print("ASIN não encontrado na URL:", url)
            return None
        asin = asin_match.group(1)
    else:
        asin = asin_match.group(2)
    hostname = urlparse(url).hostname
    if not hostname:
        print("Hostname não encontrado na URL:", url)
        return None

    country_map = {
        "amazon.com.br": "BR",
        "amazon.com": "US",
        "amazon.co.uk": "GB",
        "amazon.de": "DE",
        "amazon.ca": "CA",
        "amazon.fr": "FR",
        "amazon.es": "ES",
        "amazon.it": "IT",
        "amazon.co.jp": "JP",
        "amazon.in": "IN",
        "amazon.com.mx": "MX",
        "amazon.com.au": "AU",
    }
    country = next((country_map[key] for key in country_map if key in hostname), "US")
    return {"asin": asin, "country": country}


# --- Agente 2: Coletor de Detalhes do Produto (sem alterações) ---
def get_product_details(asin: str, country: str) -> dict:
    """Busca detalhes de um produto na Amazon usando a API da RapidAPI."""
    api_url = "https://real-time-amazon-data.p.rapidapi.com/product-details"
    querystring = {"asin": asin, "country": country}
    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": "real-time-amazon-data.p.rapidapi.com",
    }
    try:
        response = requests.get(
            api_url, headers=headers, params=querystring, timeout=30
        )
        response.raise_for_status()
        data = response.json().get("data")
        if not data:
            raise HTTPException(
                status_code=404,
                detail="Produto não encontrado na API da Amazon ou dados indisponíveis.",
            )
        return data
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503, detail=f"Erro ao chamar a API da Amazon: {e}"
        )


# --- Agente 3: Analisador com Gemini (IA) ---
# --- CORREÇÃO 2: Lógica multimodal implementada corretamente ---
def analyze_product_with_gemini(product_data: dict) -> str:
    """Usa a IA do Gemini para analisar inconsistências entre texto e imagens."""
    if not product_data:
        return "Não foi possível obter os dados do produto para análise."

    # Lógica de extração de dimensões idêntica à do Colab
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
        "Você é um analista de controle de qualidade de e-commerce extremamente meticuloso.",
        "Sua principal tarefa é encontrar inconsistências factuais entre a descrição textual de um produto e suas imagens.",
        "Analise o texto e as imagens a seguir. Aponte QUALQUER discrepância, por menor que seja, entre o que está escrito e o que é mostrado. Preste atenção especial aos dados estruturados, como dimensões, peso, voltagem, cor, material e quantidade de itens.",
        "Se tudo estiver consistente, declare: 'Nenhuma inconsistência encontrada.'",
        "\n--- DADOS TEXTUAIS DO PRODUTO ---",
        f"**Título:** {title}",
        f"**Destaques (Sobre este item):**\n- {features}",
        f"**Dimensões do Produto (extraído da tabela de especificações):** {product_dimensions_text}",
        "\n--- IMAGENS PARA ANÁLISE VISUAL ---",
    ]

    # Lógica para baixar as imagens e adicioná-las ao prompt
    image_count = 0
    for url in image_urls[:5]:  # Limita a 5 imagens para não sobrecarregar a API
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
        # A inicialização do modelo está correta, o problema era a configuração/versão
        # Usando gemini-1.5-flash para um bom custo-benefício, ou mantenha o pro
        model = genai.GenerativeModel("gemini-1.5-flash-latest")

        # A chamada generate_content agora envia texto e imagens
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Erro ao chamar a API do Gemini: {e}"
        )


# --- Endpoint Principal da API (sem alterações) ---
@app.post("/analyze", response_model=AnalyzeResponse)
def run_analysis_pipeline(request: AnalyzeRequest):
    """
    Recebe uma URL da Amazon, executa a pipeline de análise e retorna o relatório.
    """
    amazon_url = str(request.amazon_url)

    # Agente 1
    url_info = extract_product_info_from_url(amazon_url)
    if not url_info:
        raise HTTPException(
            status_code=400,
            detail="URL inválida ou ASIN não encontrado. Verifique a URL do produto.",
        )

    # Agente 2
    product_data = get_product_details(url_info["asin"], url_info["country"])

    # Agente 3
    analysis_report = analyze_product_with_gemini(product_data)

    # Extrai a imagem principal e as fotos
    main_image_url = (
        product_data.get("product_main_image_url")
        or product_data.get("main_image_url")
        or (product_data.get("product_photos") or [None])[0]
    )
    product_photos = product_data.get("product_photos") or []

    return AnalyzeResponse(
        report=analysis_report,
        asin=url_info["asin"],
        country=url_info["country"],
        product_image_url=main_image_url,
        product_photos=product_photos,
    )
