# LTC-Inspired Adaptive Leak Roadmap

## Цель

Внедрить в Genesis только одну заимствованную из Liquid Time-constant Networks идею:

- адаптивный временной масштаб мембраны через `controlled adaptive leak`

При этом мы **не** внедряем:

- ODE-решатели
- непрерывную `tau = f(x)` в стиле оригинального LTC
- новые дорогие нелинейности в hot loop
- отдельную новую архитектуру нейрона
- per-dendrite liquid gating на первом этапе

Итоговый результат должен остаться совместимым с текущими инвариантами Genesis:

- детерминизм CPU/GPU
- integer physics
- lockstep execution
- branchless или почти branchless hot loop
- LUT-driven типизация через `VariantParameters`

---

## Зачем мы это делаем

Сейчас у Genesis у каждого типа нейрона есть фиксированный `leak_rate`. Это хорошо для:

- детерминизма
- простоты GPU-ядра
- предсказуемости поведения

Но это же создает ограничение:

- нейрон живет только в одном временном масштабе
- одна и та же мембранная динамика применяется и в спокойном режиме, и при сильном сигнале, и при награде, и при шуме
- системе труднее одновременно быть и устойчивой к шуму, и быстро реагировать на важные изменения

Именно это мы и хотим улучшить.

### Проблема, которую решаем

Мы не решаем абстрактную задачу "сделать Genesis похожим на LTC". Мы решаем более приземленную инженерную проблему:

- **фиксированный leak плохо адаптируется к разным режимам среды**

На практике это означает:

- если leak слишком медленный, нейрон становится инерционным и может поздно реагировать
- если leak слишком быстрый, нейрон теряет полезное накопление сигнала и становится нервным
- один статический компромисс может быть нормальным для одной сцены, но плохим для другой

Особенно это важно для задач вроде `CartPole`, где:

- есть быстрые изменения состояния
- есть шум и дрожание сигнала
- есть reward-driven обучение через дофамин

### Наша гипотеза

Если дать нейрону **ограниченную и управляемую** адаптацию leak, то система сможет:

- быстрее реагировать в моменты, когда это важно
- дольше удерживать интеграцию, когда сигнал слабый, но значимый
- лучше переносить шумные или нестационарные входы
- делать это без отказа от integer physics и текущего hot loop

---

## Какие цели мы преследуем

### Главная цель

Сделать мембранную динамику немного более контекстно-зависимой, **не ломая архитектуру Genesis**.

### Практические цели

1. Повысить устойчивость на шумных входах.
2. Улучшить реакцию на короткие, но значимые события.
3. Снизить зависимость от одного "идеального" статического `leak_rate`.
4. Проверить, можно ли получить часть преимуществ LTC без внедрения ODE и сложной continuous-time математики.

### Архитектурные цели

1. Сохранить детерминизм CPU/GPU.
2. Не раздувать hot loop.
3. Не вводить новую тяжелую модель нейрона.
4. Сохранить объяснимость причин спайка и динамики обучения.

---

## Что мы ожидаем получить

Мы ожидаем не "магический скачок интеллекта", а конкретные локальные улучшения.

### Ожидаемые положительные эффекты

- более гибкий баланс между инерцией и чувствительностью мембраны
- меньшее число ситуаций, где один статический `leak_rate` оказывается неудачным компромиссом
- лучшая устойчивость на шумных сенсорных входах
- более естественная связка между reward-сигналом и динамикой реакции нейрона

### Что считаем успехом

Фича считается полезной, если в controlled benchmark она дает хотя бы часть следующих эффектов:

- `CartPole` обучается стабильнее
- reward variance между запусками не растет или уменьшается
- на шумных входах деградация меньше, чем у baseline
- система быстрее выходит в рабочий режим после возмущения
- при этом batch latency почти не меняется

### Что мы не ожидаем

Мы **не** ожидаем, что adaptive leak:

- заменит необходимость настраивать GSOP
- резко сократит размер сети
- автоматически улучшит все задачи
- превратит Genesis в аналог LTC/CfC

Это не новая парадигма, а точечное усиление существующей мембранной модели.

---

## Почему это стоит пробовать именно сейчас

Есть три причины, почему эта идея подходит именно текущей архитектуре:

1. **У нас уже есть дешевые источники модуляции.**
   Дофамин и `burst_count` уже существуют в системе и не требуют новой дорогой инфраструктуры.

2. **У нас уже есть LUT-модель типов нейрона.**
   Значит adaptive leak можно встроить как расширение текущей схемы `VariantParameters`, а не как отдельный слой архитектуры.

3. **У нас есть простой прикладной benchmark.**
   `CartPole` подходит как быстрый A/B стенд, на котором можно проверить, есть ли реальная польза.

---

## Что именно вводим

### Каноническая идея

Вместо фиксированного `leak_rate` вводится **эффективный leak**, который может меняться в ограниченных рамках:

```text
effective_leak = f(base_leak_rate, modulation_source, modulation_strength)
```

На первом этапе modulation source допускается только из уже существующих и дешевых сигналов:

- глобальный дофамин
- `burst_count`
- опционально локальный activity trace в более позднем milestone

### Архитектурное решение

Базовый подход:

1. `base_leak_rate` остается частью `VariantParameters`
2. вводится небольшой целочисленный модификатор
3. итоговый `effective_leak` вычисляется в integer-friendly форме
4. итоговая динамика остается дискретной

### Что считаем успешным переносом идеи LTC

Успехом считается не "реализация LTC", а следующее:

- нейрон может ускорять или замедлять утечку в зависимости от контекста
- поведение становится более адаптивным на шумных и нестационарных входах
- hot loop почти не дорожает
- проект не теряет инженерную прозрачность

---

## Scope

### In scope

- adaptive leak от дофамина
- adaptive leak от `burst_count`
- дискретные режимы мембраны
- A/B benchmark на `CartPole`
- телеметрия для новой динамики
- feature flag для быстрого отката

### Out of scope

- full LTC math
- continuous-time solver
- CfC
- новые типы дендритов под gating
- sigmoids/tanh в GPU hot path
- одновременная перестройка GSOP

---

## Технический принцип

### Базовая формула v1

Для первой версии выбираем только простую branch-friendly схему:

```text
effective_leak = clamp(base_leak_rate + leak_mod, leak_min, leak_max)
```

Где:

- `base_leak_rate` берется из `VariantParameters`
- `leak_mod` вычисляется из малого числа integer сигналов
- `clamp` остается целочисленным и детерминированным

### Источники модуляции по порядку внедрения

#### Source A: Dopamine leak modulation

```text
leak_mod = (dopamine * dopamine_leak_gain) >> shift
```

Свойства:

- уже есть в runtime
- биологически интерпретируемо
- почти нулевой memory overhead
- легко включать и выключать

#### Source B: Burst-driven leak modulation

```text
leak_mod = burst_count * burst_leak_gain
```

Свойства:

- использует уже существующий `burst_count`
- делает временную динамику activity-dependent
- не требует нового per-neuron state

#### Source C: Discrete membrane modes

Не continuous `tau(x)`, а небольшое число режимов:

- `stable`
- `responsive`
- `excited`
- `recovery`

Режим может определяться комбинацией:

- dopamine band
- burst band
- type profile

---

## Дорожная карта

## Milestone 0. Design Freeze [Design Freeze: DONE]

### Цель

Зафиксировать минимальный объём внедрения и не дать проекту расползтись в "почти LTC".

### Deliverables

- утвержденный scope этого документа
- список invariants, которые нельзя нарушать
- список запрещенных решений для v1

### Invariants (нельзя нарушать)

- **Детерминизм:** одинаковый `master_seed` и одинаковая конфигурация должны давать воспроизводимый результат на CPU и GPU.
- **Integer physics:** мембранная динамика GLIF и leak-модуляция остаются целочисленными; float-арифметика в hot loop не допускается.
- **Lockstep execution:** Day Phase, batch boundaries и текущая модель синхронизации не меняются.
- **SoA / VRAM layout:** существующая раскладка `ShardVramPtrs` и состояние шарда не ломаются ради adaptive leak.
- **LUT-driven typing:** типизация по `VariantParameters` сохраняется; новые классы нейронов не вводятся.
- **C-ABI contract:** при `adaptive_leak = false` бинарные контракты, загрузка `.state`/`.axons` и поведение baseline должны оставаться совместимыми.
- **Branch-minimal runtime:** hot loop остается branchless или почти branchless; heavy control flow в критическом пути запрещен.

### Запрещенные решения для v1

- ODE-решатели и continuous-time интеграция.
- Непрерывная `tau = f(x)` от полной дендритной суммы.
- `sigmoid`, `tanh` и другие дорогие нелинейности в GPU hot path.
- Новые per-neuron state arrays только ради поддержки adaptive leak.
- Изменение GSOP в том же milestone, где вводится adaptive leak.
- Per-dendrite или per-synapse gating.
- Полная математика LTC / CfC вместо ограниченной leak-модуляции.

### Acceptance criteria

- команда согласна, что внедряется только `controlled adaptive leak`
- зафиксировано, что GSOP не меняется в том же milestone
- зафиксировано, что full LTC и ODE не рассматриваются

### Выходные артефакты

- этот файл
- короткая ссылка на него из архитектурной документации при необходимости

---

## Milestone 1. Core Formula and ABI Preparation

### Цель

Подготовить ядро модели и C-ABI к adaptive leak без изменения поведения по умолчанию.

### Задачи

1. Расширить `VariantParameters` новыми полями для модуляции leak.
2. Сохранить backward-compatible default values.
3. Обновить CPU-side расчётную модель в `genesis-core`.
4. Обновить документацию структуры `VariantParameters`.
5. Подготовить feature flag, например:

```text
adaptive_leak = false
```

### Предлагаемые поля v1

- `adaptive_leak_mode: u8`
- `dopamine_leak_gain: i16`
- `burst_leak_gain: i16`
- `leak_min: i16`
- `leak_max: i16`

Если ABI size критичен, допускается упаковка или reuse существующего резерва, но только без нарушения бинарного контракта.

### Точки изменения

- `genesis-core`:
  - layout / physics / конфиг-парсинг
- `genesis-compute`:
  - `ffi.rs`
  - constant memory upload path
- docs:
  - `docs/specs/03_neuron_model.md`
  - `docs/specs/05_signal_physics.md`
  - `docs/specs/07_gpu_runtime.md`

### Acceptance criteria

- при `adaptive_leak = false` поведение бит-в-бит совпадает с текущим
- сборка проходит на CUDA и mock path
- бинарный контракт явно задокументирован

---

## Milestone 2. Dopamine-Driven Adaptive Leak [DONE]

### Цель

Внедрить самый дешёвый и безопасный вариант: leak, зависящий от дофамина.

### Задачи

1. Добавить integer-модуляцию leak в `compute_glif()` и GPU `UpdateNeurons`.
2. Ограничить модификацию через `leak_min` / `leak_max`.
3. Сделать поведение полностью branch-safe.
4. Добавить runtime toggle на уровне конфига типа нейрона.
5. Добавить телеметрию:
   - средний effective leak
   - количество нейронов в modulated state

### Формула v1

```text
leak_mod = (current_dopamine * dopamine_leak_gain) >> 7
effective_leak = clamp(base_leak_rate + leak_mod, leak_min, leak_max)
```

### Риски

- слишком сильная модуляция разрушит уже настроенную мембранную динамику
- влияние дофамина станет двойным: и на plasticity, и на time-scale
- может ухудшиться интерпретируемость причин обучения

### Меры защиты

- маленькие дефолтные gain
- узкий clamp window
- feature flag off by default
- отдельные A/B прогоны без изменения GSOP-констант

### Acceptance criteria

- runtime остается стабильным
- fps / ticks / batch latency не деградируют заметно
- на отключенном флаге поведение идентично baseline
- на включенном флаге эффект виден в телеметрии

---

## Milestone 3. Burst-Driven Adaptive Leak

### Цель

Добавить activity-dependent modulation без нового состояния нейрона.

### Задачи

1. Использовать существующий `burst_count` как источник модуляции leak.
2. Сделать burst modulation независимой или комбинируемой с dopamine modulation.
3. Проверить, не возникает ли runaway effect у часто спайкующих нейронов.

### Формула v1

```text
leak_mod = burst_count * burst_leak_gain
effective_leak = clamp(base_leak_rate + leak_mod, leak_min, leak_max)
```

### Риски

- self-amplifying feedback loop
- увеличение чувствительности к шуму
- конфликт с текущей логикой BDP

### Меры защиты

- hard clamp
- ограничение `burst_leak_gain`
- отдельные тесты на спайковый шторм

### Acceptance criteria

- нет эпилептической самораскачки на базовых сценариях
- BDP сохраняет ожидаемое поведение
- burst modulation даёт предсказуемый эффект на small benchmark

---

## Milestone 4. Discrete Membrane Modes

### Цель

Добавить интерпретируемую "жидкость" через дискретные режимы, а не через сложную непрерывную формулу.

### Задачи

1. Определить 3-4 канонических режима мембраны.
2. Описать правила переходов между режимами.
3. Реализовать режимы через LUT-friendly thresholds.
4. Добавить отображение режима в телеметрию/дашборд.

### Предлагаемые режимы

- `stable`: базовый leak
- `responsive`: leak слегка уменьшен, интеграция дольше
- `excited`: leak выше, быстрый отклик
- `recovery`: leak повышен, система быстрее сбрасывает накопление

### Риски

- режимов станет слишком много
- логика переходов станет менее прозрачной, чем сама формула leak

### Меры защиты

- максимум 4 режима
- режимы выводятся в телеметрию
- никакой скрытой автоматики без логируемых переходов

### Acceptance criteria

- режимы понятны человеку по логам
- режимы не требуют дорогостоящих вычислений
- поведение на шуме становится либо устойчивее, либо быстрее стабилизируется

---

## Milestone 5. Benchmark and Validation on CartPole

### Цель

Проверить, нужен ли adaptive leak на практике.

### Основной принцип

Никаких "ощущений". Только сравнительные метрики.

### Эксперименты

1. Baseline:
   - текущий Genesis без adaptive leak
2. Adaptive leak from dopamine only
3. Adaptive leak from burst only
4. Combined modulation
5. Combined modulation + noise in inputs

### Метрики

- средняя длина эпизода
- скорость обучения до целевого порога
- устойчивость к шуму входов
- средний spike rate
- доля насыщенных весов
- средний effective leak
- variance reward по seed

### Обязательные условия

- фиксированный `master_seed`
- одинаковые batch settings
- одинаковые GSOP-параметры между прогонами
- отдельное логирование конфигурации модуляции

### Acceptance criteria

- хотя бы один режим adaptive leak показывает измеримую пользу
- нет критического ухудшения стабильности
- overhead hot loop остается приемлемым

---

## Milestone 6. Hardening and Documentation

### Цель

Закрепить решение только если оно прошло benchmark и не ломает архитектуру.

### Задачи

1. Почистить временный экспериментальный код.
2. Задокументировать final contract.
3. Добавить troubleshooting section.
4. Добавить recommended defaults.
5. Оставить простой способ полного отключения.

### Документы на обновление

- `docs/specs/03_neuron_model.md`
- `docs/specs/05_signal_physics.md`
- `docs/specs/07_gpu_runtime.md`
- `docs/Architecture_and_Troubleshooting.md`

### Acceptance criteria

- у функции есть документированный default-off или safe-default режим
- новые поля и их диапазоны задокументированы
- типичный ML engineer может включить feature без чтения исходников GPU kernels

---

## Файл-уровневый план внедрения

### `genesis-core`

#### Изменить

- модели `VariantParameters`
- конфиг-парсинг типов нейрона
- `compute_glif()`
- unit tests на integer leak modulation

#### Добавить тесты

- clamp correctness
- deterministic equality CPU/GPU
- dopamine gain bounds
- burst gain bounds

### `genesis-compute`

#### Изменить

- `ffi.rs`
- upload constant memory path
- CUDA/HIP `UpdateNeurons`
- mock backend

#### Добавить тесты

- adaptive leak disabled equals baseline
- adaptive leak enabled changes only expected arithmetic path
- no overflow / underflow outside clamp

### `genesis-node`

#### Изменить

- telemetry export для effective leak / mode counters
- при необходимости CLI flag для experiment mode

### `examples/cartpole`

#### Изменить

- конфиг запуска benchmark
- noise injection сценарии
- фиксация режимов прогона

---

## Тестовая стратегия

## Unit tests

- `compute_glif()` с adaptive leak off
- `compute_glif()` с dopamine modulation
- `compute_glif()` с burst modulation
- `clamp` на крайних значениях

## Integration tests

- один shard, deterministic replay
- сравнение CPU reference и GPU result
- проверка совместимости с существующей plasticity

## Performance tests

- latency `UpdateNeurons`
- влияние на batch time
- влияние на occupancy / throughput

## Behavioral tests

- baseline CartPole
- noisy CartPole
- unstable reward schedule

---

## Риски и правила остановки

## Красные флаги

- hot loop заметно замедляется
- adaptive leak делает поведение менее объяснимым без прироста качества
- возникает сильная зависимость от тонкой подстройки gain
- результаты нестабильны между seed even under fixed setup
- новая динамика конфликтует с GSOP сильнее, чем помогает

## Stop criteria

Работа останавливается, если после Milestone 5 выполняется хотя бы одно:

1. Нет измеримого улучшения ни на одной целевой метрике.
2. Усложнение ядра непропорционально выигрышу.
3. Требуются дополнительные state arrays ради поддержки идеи.
4. Для устойчивости приходится вводить слишком много специальных правил.

---

## Definition of Done

Фича считается завершённой только если:

1. Есть production-safe режим с минимальным overhead.
2. Есть benchmark evidence, что adaptive leak полезен.
3. Документация обновлена.
4. Есть простой rollback path.
5. Внедрение не нарушает базовые инварианты Genesis.

---

## Краткое решение

Мы внедряем не LTC как модель, а только:

- `controlled adaptive leak`
- сначала от дофамина
- затем от `burst_count`
- затем, при подтвержденной пользе, дискретные режимы мембраны

Мы **не** внедряем:

- ODE
- CfC
- continuous `tau(x)`
- сложный gating по дендритам

Именно это является целевым, реалистичным и инженерно совместимым роадмапом для Genesis.
