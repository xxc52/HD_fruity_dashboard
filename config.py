"""
Dashboard Configuration
=======================
발주의뢰 대시보드 설정
"""

# 점포 정보
STORES = {
    '210': '210-본점',
    '220': '220-무역센터점',
    '480': '480-판교점'
}

# 부서 정보
DEPARTMENTS = {
    '520': '520-식품팀'
}

PARTS = {
    '201': '201-신선가공'
}

PC_CODES = {
    '311': '311-농산물'
}

CORNERS = {
    '002': '002-청과'
}

ORDER_TYPES = {
    '07': '07-신선식품'
}

# Top 10 SKU (210 점포 기준, TSALE_AMT 합계)
TOP_10_SKUS = [
    {'code': '269211', 'name': '특선바나나', 'category': '과일', 'unit': 'EA', 'total_sales': 1_100_366_554},
    {'code': '202309', 'name': '국산청포도', 'category': '과일', 'unit': 'BOX', 'total_sales': 937_518_004},
    {'code': '400189', 'name': '금실딸기(500g/팩)', 'category': '과일', 'unit': 'EA', 'total_sales': 707_107_662},
    {'code': '400220', 'name': '죽향 딸기(500g/팩)', 'category': '과일', 'unit': 'EA', 'total_sales': 635_086_300},
    {'code': '400053', 'name': '사과(4입/봉(40상))', 'category': '과일', 'unit': 'EA', 'total_sales': 632_405_800},
    {'code': '400293', 'name': '블루베리(200g/팩)', 'category': '과일', 'unit': 'EA', 'total_sales': 550_938_300},
    {'code': '400342', 'name': '국내산 블루베리(100g/팩)', 'category': '과일', 'unit': 'EA', 'total_sales': 466_608_452},
    {'code': '202284', 'name': '불로초감귤', 'category': '과일', 'unit': 'BOX', 'total_sales': 460_417_629},
    {'code': '202193', 'name': '사과', 'category': '과일', 'unit': 'BOX', 'total_sales': 393_652_575},
    {'code': '202599', 'name': '골드키위', 'category': '과일', 'unit': 'EA', 'total_sales': 360_282_736},
]

# 예측 신뢰구간
CONFIDENCE_INTERVAL = 0.05  # ±5%
