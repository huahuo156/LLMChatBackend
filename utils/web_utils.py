import os

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from tavily import TavilyClient


def get_tavily_client():
    # 1. 从环境变量中读取API密钥
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return None

    # 2. 初始化Tavily客户端
    tavily = TavilyClient(api_key=api_key)

    return tavily


@tool
def web_search(query: str) -> str:
    """
    一个专门用于 llm agent 联网查询的通用工具，查询与参数 query 相关的内容，并返回查询结果，除非配置错误
    :param query: 必需的参数，用于描述需要联网查询的内容
    :return: 返回联网查询 query 的内容
    """

    tavily = get_tavily_client()

    if not tavily:
        return "错误：配置 tavily client 失败"

    try:
        response = tavily.search(
            query=query,
            search_depth="basic",
            include_answer=True
        )
        if response.get("answer"):
            return response["answer"]

        # 如果没有综合性回答，则格式化原始结果
        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result['title']}: {result['content']}")

        if not formatted_results:
            return f"抱歉，没有找到与{query}相关的内容。"

        return "根据搜索，为您找到以下信息：\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误：执行Tavily搜索时出现问题 - {e}"


@tool
def crawl_url_content(url: str) -> str:
    """
    一个专门用于 llm agent 联网访问指定URL的通用工具，可以对指定的 URL 进行简单的内容爬取；该工具大概率可能失败...
    :param url: 必需的参数，指定要爬取的网页
    :return: 返回爬取指定网页的内容
    """

    tavily = get_tavily_client()

    if not tavily:
        return "错误：配置 tavily client 失败"

    try:
        response = tavily.crawl(
            url=url,
            instructions="Find all pages on agents",
            max_depth=4,
            max_breadth=20,
            extract_depth="advanced"
        )

        formatted_results = []
        for result in response.get("results", []):
            formatted_results.append(f"- {result.get('title', 'None')}: {result.get('content', 'None')}")

        if not formatted_results:
            return f"抱歉，无法获取指定网页 {url} 上的内容。"

        return "根据访问，为您找到以下信息：\n" + "\n".join(formatted_results)

    except Exception as e:
        return f"错误：执行Tavily网页爬取时出现问题 - {e}"


@tool
def fetch_url_content(url: str):
    """
    爬取网页并过滤信息。
    :param url: 目标网站的URL
    :return: 过滤后的文本内容列表
    """
    # --- 默认配置 ---
    # 默认尝试查找常见的内容容器
    content_selectors = ['.main-content', 'article', '.post-body', 'p']

    # 默认过滤掉常见的无关元素
    unwanted_selectors = ['nav', '.advertisement', '.sidebar', '#footer', 'script', 'style']

    # 默认请求头，模拟浏览器
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8, application/json',
        'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        'Accept-Encoding': 'gzip, deflate, br',
        'DNT': '1',  # 表示不跟踪
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    # --- 配置结束 ---

    try:
        print(f"正在请求网页: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        response.encoding = response.apparent_encoding

        print("网页请求成功，正在解析内容...")
        soup = BeautifulSoup(response.text, 'html.parser')

        # 移除不需要的元素
        for selector in unwanted_selectors:
            for tag in soup.select(selector):
                tag.decompose()

        # 查找并提取所需内容
        content_elements = []
        for selector in content_selectors:
            elements = soup.select(selector)
            if elements:
                content_elements.extend(elements)
                break  # 找到匹配的就跳出，避免重复添加

        if not content_elements:
            print(f"警告: 在 {url} 中未找到匹配选择器 '{content_selectors}' 的内容。")
            return []

        # 提取文本内容
        extracted_texts = []
        for element in content_elements:
            text = element.get_text(strip=True)
            if text:
                extracted_texts.append(text)

        print("内容提取完成。")
        return extracted_texts

    except requests.exceptions.RequestException as e:
        print(f"请求错误: {e}")
        return []
    except Exception as e:
        print(f"解析错误或其它错误: {e}")
        return []


if __name__ == "__main__":
    res = fetch_url_content('https://baike.baidu.com/item/JaVa/85979')
    print(res)
