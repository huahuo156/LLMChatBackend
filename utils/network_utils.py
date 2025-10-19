import re

import requests
from bs4 import BeautifulSoup
from langchain_core.tools import tool


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
