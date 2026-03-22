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
    def __init__(self, control: GenesisControl, target_score: float = 400.0, window_size: int = 15, **kwargs):
        self.control = control
        # [DOD FIX] Explicit initialization for better type inference
        self.default_target = target_score
        self.window = deque(maxlen=window_size)
        self.phase = Phase.EXPLORATION
        
        self.target_score = 400.0

        # Конфигурация параметров по фазам
        self.explore_params = self._extract_params(kwargs, "explore_")
        self.distill_params = self._extract_params(kwargs, "distill_")
        self.crystallized_params = self._extract_params(kwargs, "crystallized_")

        # Инициализация фазы эксплорации
        self._apply_phase_settings(self.explore_params)

    def _extract_params(self, kwargs: dict, prefix: str) -> dict:
        """Извлекает ВСЕ параметры с данным префиксом. Любой параметр по умолчанию None."""
        params = {}
        for key, value in kwargs.items():
            if key.startswith(prefix):
                param_name = key[len(prefix):]
                params[param_name] = value
        # target_score -> target (для обратной совместимости)
        if "target_score" in params:
            params["target"] = params.pop("target_score")
        if "target" not in params:
            params["target"] = self.default_target
        return params

    def _apply_phase_settings(self, p: dict):
        """Пробрасывает параметры в манифест. Любой параметр может быть None — просто пропускается."""
        if p.get("prune") is not None:
            self.control.set_prune_threshold(p["prune"])
        if p.get("night") is not None:
            self.control.set_night_interval(p["night"])
        if p.get("sprouts") is not None:
            self.control.set_max_sprouts(p["sprouts"])
        
        # Рецепторы (variant 0/1)
        if p.get("d1") is not None and p.get("d2") is not None:
            self.control.set_dopamine_receptors(0, p["d1"], p["d2"])
            self.control.set_dopamine_receptors(1, p["d1"], p["d2"])
        
        # Физика
        if p.get("leak") is not None and p.get("homeos_penalty") is not None and p.get("homeos_decay") is not None:
            self.control.set_membrane_physics(0, p["leak"], p["homeos_penalty"], p["homeos_decay"])
            self.control.set_membrane_physics(1, int(p["leak"] * 1.5), int(p["homeos_penalty"] * 0.8), p["homeos_decay"])

        # [DOD FIX] Плоское кэширование переменных для O(1) доступа в Hot Loop
        if p.get("target") is not None: self.target_score = p["target"]


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
            # Сеть нащупала решение (например, стабильно 70% от таршета)
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
        print("\n🔥 [AutoTuner] Переход в DISTILLATION: Выжигаем слабые связи и уточняем физику...")
        self._apply_phase_settings(self.distill_params)
        self.phase = Phase.DISTILLATION

    def _transition_to_crystallization(self):
        print("\n❄️ [AutoTuner] Переход в CRYSTALLIZED: Граф заморожен. Идеальный навык.")
        self._apply_phase_settings(self.crystallized_params)
        
        # [DOD FIX] Аппаратное отключение электрической пластичности (GSOP)
        self.control.disable_all_plasticity() 
        self.phase = Phase.CRYSTALLIZED

    def _transition_to_exploration(self):
        print("\n🌱 [AutoTuner] Откат в EXPLORATION: Навык утерян, возобновляем рост...")
        self._apply_phase_settings(self.explore_params)
        self.phase = Phase.EXPLORATION
        self.window.clear() # Сбрасываем окно, чтобы не прыгать туда-сюда
