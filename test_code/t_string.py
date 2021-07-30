

search_str = "메리츠 화재 삼성화재 쿠팡"

synonym_str ="메리츠 보험 회사"

for x in search_str.split():
   print("True") if x in synonym_str else print("False")