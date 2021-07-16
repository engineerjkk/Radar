# Radar 07-16

## 1.로그파일 파싱.py 
[Raw데이터 Parsing 파일](https://github.com/engineerjkk/Radar/blob/main/PassingOK_NotRealTime_mmw_demo_example_script.py)  
* SRS로부터 저장된 Raw 데이터를 parsing해서 csv파일로 저장한다.  

### 핵심

[mmw_demo_example_script.py](https://github.com/engineerjkk/Radar/blob/main/PassingOK_NotRealTime_mmw_demo_example_script.py)안의  
detectedNoise_array = parser_one_mmw_demo_output_packet(**allBinData[totalBytesParsed::1]**, readNumBytes-totalBytesParsed)  
여기의 allBinData 데이터 형식이 어떻게 되는지 분석해야한다.
 
### 시퀀스
같은 폴더내에 저장된 crd 파일을 불러온다. 이때 실행하는 방법은 다음과 같다.

1. 명령프롬프트 창에서  cd.. 를 반복해 가장 base 단으로 옮긴뒤 파이썬 파일이 담긴 경로를 복사해준다.  
2. 그 다음 바로 이어서 srs_test_1.crd 를 입력해주면된다.

나는 파이참 터미널 창에서 실행하였다.
1. 먼저 다음과 같은 위치 폴더에 crd 파일(SRS로부터 로그한 Raw파일)을 저장해준다.
 ![image](https://user-images.githubusercontent.com/76835313/125942277-6f479924-c51a-4cfc-aa05-af3150b91c6e.png)
2. 파이참 터미널 환경에서 다음과 같이 작성한다.
![image](https://user-images.githubusercontent.com/76835313/125942395-604fd673-eb94-41f9-ae12-b2239a41b49a.png)
(base) I:\junekoo.kang\subin\parser_scripts>python mmw_demo_example_script.py src_test_1.crd
3. 실행결과는 다음과 같다.
![image](https://user-images.githubusercontent.com/76835313/125942499-b96312ee-91b7-410b-a380-7531254b7050.png)
5. 이렇게 생성된 값들은 csv 파일 형태로 excel에 저장된다.
-> [생성된 csv파일](https://github.com/engineerjkk/Radar/blob/main/mmw_demo_output.csv)

**여기까지가 Raw데이터를 Parsing한 것이다. 하지만 문제는 실시간으로 해결 하지못하고 로그데이터를 사용했다.**

## 2. 실시간 시각화.py
1. 레이다 연결 및 Port 확인
![image](https://user-images.githubusercontent.com/76835313/125943258-0bdad3e9-be24-4f5f-a467-0452ad7f8668.png)
2. SRS 실행
![image](https://user-images.githubusercontent.com/76835313/125943337-f59f1bef-4541-4d3e-b130-9c65f6328cb7.png)
![image](https://user-images.githubusercontent.com/76835313/125943440-dc810f9b-269c-4ccd-afb0-e2857002e374.png)
3. SRS 종류 후 파이참으로 파이썬 파일 실행
[readData_IWR6843ISK_DEPRECIATED.py](https://github.com/engineerjkk/Radar/blob/main/RealTimeOK_readData_IWR6843ISK_DEPRECIATED.py)
4. 생성된 시각화. 하지만 그렇게 정확해보이진 않는다.
![image](https://user-images.githubusercontent.com/76835313/125943830-98ca5ff5-7f3c-4224-92d5-20628120c922.png)

# 결론
parsing이 되었던 코드와 실시간 시각화가 됐던 코드를 합쳐야 한다. 그래야 진짜 가공된 데이터를 실제 실시간으로 시각화하여 볼수 있다.  
그 중심에는 [Raw데이터 Parsing 파일](https://github.com/engineerjkk/Radar/blob/main/PassingOK_NotRealTime_mmw_demo_example_script.py) 코드 속의 allBinData 데이터 형식를 아는것이 핵심이다.

