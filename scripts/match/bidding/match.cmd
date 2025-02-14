set BEN_HOME=D:\github\ben\
set trust=%1
set boards=%2

python auction.py --bidderNS=%trust%.conf --bidderEW=default.conf --set=%boards% > .\%trust%\auctionsNS.json
python auction.py --bidderNS=default.conf --bidderEW=%trust%.conf --set=%boards% > .\%trust%\auctionsEW.json

type ".\%trust%\auctionsNS.json" | python lead.py --bidder=default.conf > .\%trust%\leads1.json
type ".\%trust%\auctionsEW.json" | python lead.py --bidder=%trust%.conf > .\%trust%\leads2.json

type ".\%trust%\leads1.json" | python score.py > .\%trust%\results1.json  
type ".\%trust%\leads2.json" | python score.py > .\%trust%\results2.json  

python ..\..\..\src\compare.py .\%trust%\results1.json .\%trust%\results2.json > .\%trust%\compare.json 

type ".\%trust%\compare.json" | python printmatch.py >.\%trust%\match.txt

type ".\%trust%\compare.json" | python printmatchashtml.py >.\%trust%\html\match.html

powershell -Command "Get-Content .\%trust%\compare.json | jq .imp | Measure-Object -Sum | Select-Object -ExpandProperty Sum"

type ".\%trust%\compare.json" | jq .imp >.\%trust%\imps.txt  

rem find and sum positive scores
powershell -Command "(Get-Content .\%trust%\imps.txt | ForEach-Object { $_ -as [double] }) | Where-Object { $_ -gt 0 } | Measure-Object -Sum | Select-Object -ExpandProperty Sum"

python printdeal.py %boards% %1\html