# Технічна документація ККП: Реалізація та Архітектура

У цьому файлі зібрано повний опис реалізації, архітектурні діаграми та специфікацію (SRS), які були згенеровані для проєкту "Програмна система бінарної класифікації мовного сигналу за критерієм «Людина-ШІ»".

---

## 1. Структура проєкту
Проєкт організований згідно з принципами модульності та розділення відповідальності (Separation of Concerns):
- `src/bionic_core.py` — математична реалізація біонічного алгоритму.
- `src/ml_core.py` — обгортка для класичних алгоритмів машинного навчання.
- `src/app.py` — графічний інтерфейс на базі Plotly Dash.
- `docs/` — технічна документація та UML-схеми.

---

## 2. UML Діаграми (PlantUML)

### Діаграма Прецедентів (Use Case Diagram)
```plantuml
@startuml
skinparam handwritten false
skinparam monochrome false

actor "Користувач (Дослідник)" as User

rectangle "Система класифікації мовного сигналу" {
    usecase "Завантаження WAV файлу" as UC1
    usecase "Попередня обробка сигналу" as UC2
    
    usecase "Аналіз біонічним методом" as UC3
    usecase "Побудова простору Хелвага-Щерби" as UC3_1
    usecase "Пошук центрів щільності" as UC3_2
    usecase "Обчислення відстаней R" as UC3_3
    
    usecase "Аналіз методом Machine Learning" as UC4
    usecase "Екстракція MFCC, Chroma, Rolloff" as UC4_1
    usecase "Класифікація RandomForest/SVM" as UC4_2
    
    usecase "Відображення результатів (Дашборд)" as UC5
}

User --> UC1
User --> UC5

UC1 ..> UC2 : <<include>>
UC2 ..> UC3 : <<include>>
UC2 ..> UC4 : <<include>>

UC3 ..> UC3_1 : <<include>>
UC3 ..> UC3_2 : <<include>>
UC3 ..> UC3_3 : <<include>>

UC4 ..> UC4_1 : <<include>>
UC4 ..> UC4_2 : <<include>>

UC3 --> UC5 : Передає вердикт
UC4 --> UC5 : Передає вердикт
@enduml
```

### Діаграма Компонентів (Component Diagram)
```plantuml
@startuml
skinparam componentStyle uml2

package "Frontend (UI)" {
    [Dash Dashboard\n(app.py)] as Dashboard
    [Plotly Visualization] as Plotly
}

package "Backend Core" {
    [Bionic Classifier\n(bionic_core.py)] as Bionic
    [ML Classifier\n(ml_core.py)] as ML
    [Audio Processor\n(scipy/librosa)] as Audio
}

database "File System" {
    [WAV Files] as WAV
}

actor User

User --> Dashboard : "Завантажує файл"
Dashboard --> WAV : "Тимчасове збереження"
Dashboard --> Bionic : "Виклик аналізу"
Dashboard --> ML : "Виклик аналізу"

Bionic --> Audio : "scipy.io.wavfile"
ML --> Audio : "librosa.load"

Bionic --> Plotly : "Координати центрів"
ML --> Dashboard : "Ймовірності, Клас"
@enduml
```

---

## 3. Специфікація Програмного Забезпечення (SRS)

**1. ВСТУП**
Система використовує гібридний підхід, що поєднує біонічний метод та класичне машинне навчання для ідентифікації штучно згенерованого мовлення.

**2. ФУНКЦІЇ ПРОДУКТУ**
- Завантаження WAV (22050 Hz, Mono).
- Координатно-топологічне відображення фонем.
- Класифікація за біонічним порогом (R=16).
- ML-класифікація (MFCC + Random Forest).
- Візуалізація результатів.

**3. КОНКРЕТНІ ВИМОГИ**
- **UI:** Веб-дашборд Dash з темною темою.
- **Стек:** Python 3.9+, librosa, scipy, scikit-learn.
- **Швидкодія:** Аналіз 10с аудіо < 5с.

---

## 4. Опис реалізації модулів

### Біонічне ядро (`bionic_core.py`)
Реалізує алгоритм ковзного вікна для пошуку центрів щільності точок у просторі (v1, v2). Обчислює середню відстань R між центрами. Висока варіативність (R > 16) свідчить про людське мовлення.

### ML ядро (`ml_core.py`)
Використовує бібліотеку `librosa` для отримання спектральних характеристик. MFCC коефіцієнти дозволяють вловити тонкі відмінності в тембрі та "металеві" артефакти ШІ.

### Веб-інтерфейс (`app.py`)
Забезпечує інтерактивну взаємодію. Дозволяє завантажувати файли та миттєво бачити порівняння двох методів на графіках.
