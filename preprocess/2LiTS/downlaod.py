import requests
from tqdm import tqdm
# Define the URL and headers
# url = "https://drive.usercontent.google.com/download?id=0B0vscETPGI1-TE5KWFgxaURubFE&export=download&authuser=0&resourcekey=0-0fwNqxVQJSfYDvSt1Kr_Sg&confirm=t&uuid=013dca50-943a-4122-8037-81a5e2803902&at=AO7h07c0_r3kV08Hbr4Bdrt1ciwX:1724751998934"
url = "https://drive.usercontent.google.com/download?id=0B0vscETPGI1-cTZGbTU4UC05Qm8&export=download&authuser=0&resourcekey=0-qf26pTzXmgVv_qznEBDNqQ&confirm=t&uuid=3bd9d016-4f79-49ca-9c2b-6a121bbc7d2a&at=AO7h07cxFPY34OeDu3bkh1Igb2dn:1724752247791"
headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Accept-Language": "en,zh;q=0.9,zh-CN;q=0.8",
    "Cookie": "S=billing-ui-v3=lDNe7eJ9VvRaaPneHXeO77T7kIjM989n:billing-ui-v3-efe=lDNe7eJ9VvRaaPneHXeO77T7kIjM989n; SEARCH_SAMESITE=CgQI9JsB; SID=g.a000nAiOJx_RiNuAhn3vGDtxEJgD0xW2q_9j8-mPFhYyDYQy0c7D4Qr-9XjLLQ-8JnpQQmKNGwACgYKAZ4SARQSFQHGX2Mi6xZLAeDHVvAiRIL8W3FDTxoVAUF8yKowPWal0zTTL9CsSyyecOFU0076; __Secure-1PSID=g.a000nAiOJx_RiNuAhn3vGDtxEJgD0xW2q_9j8-mPFhYyDYQy0c7D4pr1jCVd1v9QLtiN8T5z1wACgYKAZwSARQSFQHGX2Miif896V7YFlOv04EBYUtrXRoVAUF8yKrC1hQb2f9Bsy4Ig8IPWyfo0076; __Secure-3PSID=g.a000nAiOJx_RiNuAhn3vGDtxEJgD0xW2q_9j8-mPFhYyDYQy0c7DJyGCyfBAmxpX38cG8sQ8aAACgYKATQSARQSFQHGX2Miywa5D5cDtRmyf74OA9vShRoVAUF8yKrCslHmO9T5dH-Mza2aA-cu0076; HSID=AaJz_Jm42i22DX7qM; SSID=AJNn35qfQ96PQxRlk; APISID=zj6ZBXg-IoLHhly2/AjLeishafk2ItWHnJ; SAPISID=hl2MUCP8wLRYDSsI/AQF6EV929xM16gWZ6; __Secure-1PAPISID=hl2MUCP8wLRYDSsI/AQF6EV929xM16gWZ6; __Secure-3PAPISID=hl2MUCP8wLRYDSsI/AQF6EV929xM16gWZ6; AEC=AVYB7cr44X4XppbCQmP-pHQVZcpgPDD9zWa-1cykkIEryvNZVF5vG4Pg5w; NID=517=Z7dt-eEaB52pL53YQeF1bOFZNhKuSOs5pB_ieuQSgy-Ely3GYKlT_barkMJWFoaB4P-QanhGPIZ1tkCdxeQ9qk3Jm_86WlvuubZU3GaUABcz-lIKXVF5Rzy5jNVJxxforihCb2OY8HeV7BuLjqvcPsSFxsu5za9ron5Lsy47oDWAIJIcoK-mBgKcim2LLXUrMWz6c866Vwhf861qjgLQoSJdpKSqLYMbajCqRfxr562szv9uqvPFLt74eo6Vcjj69M5-uzqkFBD_BKRxBqDyq84p9yXcy0gDWpsQg03MwkgTXcOmFi3Km5fB8n2jh_4hHYFlUQX9hto5qWqmSgqsW36aB65ZTHhHrN2BnQ_m2KcoeRnLP8aQO9hRNT763CPWeEX_I5SGegZ3kVM_e7Ai_e8gb_mfde3DLampSkAt_Hm-yl2dhBQfdA; __Secure-1PSIDTS=sidts-CjIBUFGoh5gd7pWNKZmSQ5e4PVrXSzRpCH2I0Pf-knGbUlzuenSPVLmI2Ec4TZi76LCSYRAA; __Secure-3PSIDTS=sidts-CjIBUFGoh5gd7pWNKZmSQ5e4PVrXSzRpCH2I0Pf-knGbUlzuenSPVLmI2Ec4TZi76LCSYRAA; SIDCC=AKEyXzUrS-tmnZWtR7p4710ePpIEF9AO4Lxktq3rN5OY8Wq71vQndPGrcNCDs7zcTn4fvH3hTQmw; __Secure-1PSIDCC=AKEyXzXl5e8xXgoEvleUM5Qt9A0eZh6IFfTdEF2iUUnUPTY9lSHVr9Jd7Snqy0Y5RqrPSRtJpMo; __Secure-3PSIDCC=AKEyXzXkSvkSRCDpUq2NzC7aDra-RDURikZc6uzQXy0b5j28RxSQLCAaVaDNsP9kPEJJK98KVDdn",
    "Sec-CH-UA": "\"Not)A;Brand\";v=\"99\", \"Google Chrome\";v=\"127\", \"Chromium\";v=\"127\"",
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": "\"macOS\"",
    "Sec-Fetch-Dest": "iframe",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-site",
    "Upgrade-Insecure-Requests": "1"
}

# Make the GET request to download the file
response = requests.get(url, headers=headers, stream=True)

# Save the file locally
file_name = "/home/local/ASURITE/longchao/Desktop/project/GE_health/SegmentAsYouWish/data/2LiTS/Training_Batch2.zip"
with open(file_name, "wb") as file:
    for chunk in tqdm(response.iter_content(chunk_size=8192)):
        if chunk:
            file.write(chunk)

print(f"File downloaded successfully as {file_name}")
