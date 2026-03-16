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
        self.dopamine_pulse = 0
        self.dopamine_reward = 0
        self.dopamine_punish = 0
        self.err_angle_weight = 0.0
        self.err_vel_weight = 0.0
        self.angle_limit = 0.0
        self.vel_limit = 0.0
        self.shock_base = 0
        self.shock_bitshift = 0
        self.shock_vel_mult = 0
        self.shock_max_batches = 0

        # Конфигурация параметров по фазам
        self.explore_params = self._extract_params(kwargs, "explore_")
        self.distill_params = self._extract_params(kwargs, "distill_")
        self.crystallized_params = self._extract_params(kwargs, "crystallized_")

        # Инициализация фазы эксплорации
        self._apply_phase_settings(self.explore_params)

    def _extract_params(self, kwargs: dict, prefix: str) -> dict:
        """Извлекает набор параметров для конкретной фазы с дефолтными значениями."""
        is_explore = prefix == "explore_"
        return {
            "target": kwargs.get(f"{prefix}target_score", self.default_target),
            "prune": kwargs.get(f"{prefix}prune", 30 if is_explore else (150 if prefix == "distill_" else 0)),
            "night": kwargs.get(f"{prefix}night", 10000 if is_explore else (5000 if prefix == "distill_" else 0)),
            "sprouts": kwargs.get(f"{prefix}sprouts", 128 if is_explore else (2 if prefix == "distill_" else 0)),
            
            # Баланс R-STDP
            "dopamine_pulse": kwargs.get(f"{prefix}dopamine_pulse", -1),
            "dopamine_reward": kwargs.get(f"{prefix}dopamine_reward", 15),
            "dopamine_punish": kwargs.get(f"{prefix}dopamine_punish", -210),
            
            # Физика мембраны
            "leak": kwargs.get(f"{prefix}leak", 850),
            "homeos_penalty": kwargs.get(f"{prefix}homeos_penalty", 5560),
            "homeos_decay": kwargs.get(f"{prefix}homeos_decay", 49),
            
            # Рецепторы
            "d1": kwargs.get(f"{prefix}d1", 172),
            "d2": kwargs.get(f"{prefix}d2", 252),
            
            # Градиент
            "err_angle": kwargs.get(f"{prefix}err_angle", 0.8),
            "err_vel": kwargs.get(f"{prefix}err_vel", 0.2),
            "angle_limit": kwargs.get(f"{prefix}angle_limit", 0.2094),
            "vel_limit": kwargs.get(f"{prefix}vel_limit", 2.0),

            # Шок
            "shock_base": kwargs.get(f"{prefix}shock_base", 0),
            "shock_bitshift": kwargs.get(f"{prefix}shock_bitshift", 5),
            "shock_vel_mult": kwargs.get(f"{prefix}shock_vel_mult", 5),
            "shock_max_batches": kwargs.get(f"{prefix}shock_max_batches", 2)
        }

    def _apply_phase_settings(self, p: dict):
        """Пробрасывает параметры в манифест и кэширует их для Hot Loop (Zero-Overhead)."""
        self.control.set_prune_threshold(p["prune"])
        self.control.set_night_interval(p["night"])
        self.control.set_max_sprouts(p["sprouts"])
        
        # Рецепторы (variant 0/1)
        self.control.set_dopamine_receptors(0, p["d1"], p["d2"])
        self.control.set_dopamine_receptors(1, p["d1"], p["d2"])
        
        # Физика
        self.control.set_membrane_physics(0, p["leak"], p["homeos_penalty"], p["homeos_decay"])
        self.control.set_membrane_physics(1, int(p["leak"] * 1.5), int(p["homeos_penalty"] * 0.8), p["homeos_decay"])

        # [DOD FIX] Плоское кэширование переменных для O(1) доступа в Hot Loop
        self.target_score = p["target"]
        self.dopamine_pulse = p["dopamine_pulse"]
        self.dopamine_reward = p["dopamine_reward"]
        self.dopamine_punish = p["dopamine_punish"]
        self.err_angle_weight = p["err_angle"]
        self.err_vel_weight = p["err_vel"]
        self.angle_limit = p["angle_limit"]
        self.vel_limit = p["vel_limit"]
        self.shock_base = p["shock_base"]
        self.shock_bitshift = p["shock_bitshift"]
        self.shock_vel_mult = p["shock_vel_mult"]
        self.shock_max_batches = p["shock_max_batches"]


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
