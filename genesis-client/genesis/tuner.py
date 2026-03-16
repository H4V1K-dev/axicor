from collections import deque
from enum import Enum
from .control import GenesisControl

class Phase(Enum):
    EXPLORATION = 1   # Бурный рост, много дофамина, низкий порог прунинга
    DISTILLATION = 2  # Выжигание шума, высокий порог прунинга, частый сон
    CRYSTALLIZED = 3  # Чистый инференс, сон и пластичность отключены

class GenesisAutoTuner:
    """
    Production State Machine для автоматической дистилляции топологии.
    Оперирует скользящим средним (SMA) метрик среды.
    """
    def __init__(self, control: GenesisControl, target_score: float, window_size: int = 15):
        self.control = control
        self.target_score = target_score
        self.window = deque(maxlen=window_size)
        self.phase = Phase.EXPLORATION
        
        # Инициализация фазы эксплорации
        self.control.set_prune_threshold(30)
        self.control.set_night_interval(10000)

    def step(self, episode_score: float) -> Phase:
        """
        Вызывается в конце каждого эпизода среды.
        Возвращает текущую фазу для логирования.
        """
        if self.phase == Phase.CRYSTALLIZED:
            return self.phase
            
        self.window.append(episode_score)
        if len(self.window) < self.window.maxlen:
            return self.phase # Ждем накопления статистики
            
        sma_score = sum(self.window) / len(self.window)
        
        if self.phase == Phase.EXPLORATION:
            # Сеть нащупала решение (например, стабильно 70+ очков из 100)
            if sma_score >= self.target_score * 0.7:
                self._transition_to_distillation()
                
        elif self.phase == Phase.DISTILLATION:
            # Сеть очистилась от шума и достигла идеала
            if sma_score >= self.target_score:
                self._transition_to_crystallization()
            # Сеть забыла нужный навык из-за слишком жесткого прунинга
            elif sma_score < self.target_score * 0.4:
                self._transition_to_exploration()
                
        return self.phase

    def _transition_to_distillation(self):
        print("\n🔥 [AutoTuner] Переход в DISTILLATION: Выжигаем слабые связи...")
        # Поднимаем порог удаления синапсов (выживут только самые сильные)
        self.control.set_prune_threshold(150)
        # Форсируем частые фазы сна для агрессивной пересборки графа
        self.control.set_night_interval(5000)
        self.phase = Phase.DISTILLATION

    def _transition_to_crystallization(self):
        print("\n❄️ [AutoTuner] Переход в CRYSTALLIZED: Граф заморожен. Идеальный навык.")
        # Отключаем сон и пластичность — переводим сеть в режим HFT-инференса
        self.control.set_night_interval(0)
        self.control.set_prune_threshold(0)
        self.phase = Phase.CRYSTALLIZED

    def _transition_to_exploration(self):
        print("\n🌱 [AutoTuner] Откат в EXPLORATION: Навык утерян, возобновляем рост...")
        self.control.set_prune_threshold(30)
        self.control.set_night_interval(10000)
        self.phase = Phase.EXPLORATION
        self.window.clear() # Сбрасываем окно, чтобы не прыгать туда-сюда
