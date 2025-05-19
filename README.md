# AirPiano 프로젝트 🎹

**AirPiano**는 손가락의 위치와 그림자 정보를 활용하여 실제 피아노 없이도 연주할 수 있도록 만든 가상 악기 시스템입니다.  
Mediapipe, OpenCV, Pygame을 활용해 손을 인식하고, 손가락이 바닥에 닿았는지를 그림자로 판별하여 해당 음을 재생합니다.

---

## 📁 프로젝트 구조
AirPiano/
├── main.py # 실제 연주 화면 실행용
├── istouch.py # 손가락-바닥 터치 판별 전용 모듈
├── utils.py # 거리 및 그림자 위치 계산 로직
├── assets/ # 사운드(.wav) 파일 보관 폴더
├── tests/ # 테스트 폴더
│ ├── test_main_utils.py # 유닛 테스트
│ └── test_istouch_mock.py # mock 테스트
├── 기록/ # 기록 기능 (기능 확장 모듈)
├── .gitignore # Git 무시 설정


---

## 🧪 테스트 방법

### ✅ 유닛 테스트 (utils.py 기반)
- 거리 계산, 그림자 위치 판별 등 내부 로직 검증

### ✅ 목 테스트 (istouch.py 기반)
- OpenCV의 카메라 스트리밍, 화면 출력 함수 등을 mock으로 대체하여 테스트 가능

### 실행 명령어:

```bash
python -m unittest discover -s tests

🛠 사용 기술
Python 3.10+

OpenCV

Mediapipe

Pygame

unittest, unittest.mock

✅ 브랜치 정보
main → 원본 소스

test/unittest-mock → 테스트 (unittest + mock) 적용 브랜치

💡 참고
이 프로젝트는 실제 손-그림자 인터랙션 기반 인터페이스를 탐색하기 위한 실험용 도구이며, 다양한 HCI 인터페이스나 비접촉식 입력 장치로 확장할 수 있습니다.