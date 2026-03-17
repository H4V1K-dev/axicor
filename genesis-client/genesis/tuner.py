from collections import deque
from enum import Enum
from .control import GenesisControl

class Phase(Enum):
    EXPLORATION = 1   # Бурный рост, много дофамина, низкий порог прунинга
    DISTILLATION = 2  # Выжигание шума, высокий порог прунинга, частый сон
    CRYSTALLIZED = 3  # Чистый инференс, сон и пластичность отключены

# Пороги для анализа (логирование)
EXPLORATION_PRUNE = 30
EXPLORATION_NIGHT = 10000
DISTILLATION_PRUNE = 100   # Смягчено: было 150 → меньше выжигаем важных слабых связей
DISTILLATION_NIGHT = 7500  # Реже сон: было 5000 → даём связям окрепнуть
ROLLBACK_RATIO = 0.3      # Смягчено: было 0.4 → реже откат из-за шума
DISTILLATION_ENTER_RATIO = 0.5  # SMA ≥ target * 0.5 для входа в DISTILLATION
# При забывании reward падает → больше punishment → LTD → ещё больше забывание (порочный круг).
PUNISHMENT_SOFTEN_RATIO = 0.5   # Множитель shock-батчей при обнаружении забывания
PUNISHMENT_SOFTEN_SMA_RATIO = 0.5  # SMA < target * это → смягчаем punishment

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
        self.last_sma: float = 0.0

        # Текущие параметры рантайма (для логирования и анализа)
        self.current_prune_threshold: int = EXPLORATION_PRUNE
        self.current_night_interval: int = EXPLORATION_NIGHT

        # Инициализация фазы эксплорации
        self.control.set_prune_threshold(EXPLORATION_PRUNE)
        self.control.set_night_interval(EXPLORATION_NIGHT)

    def step(self, episode_score: float) -> Phase:
        """
        Вызывается в конце каждого эпизода среды.
        Возвращает текущую фазу для логирования.
        """
        if self.phase == Phase.CRYSTALLIZED:
            return self.phase

        self.window.append(episode_score)
        if len(self.window) < self.window.maxlen:
            self.last_sma = sum(self.window) / len(self.window) if self.window else 0.0
            return self.phase  # Ждем накопления статистики

        sma_score = sum(self.window) / len(self.window)
        self.last_sma = sma_score

        if self.phase == Phase.EXPLORATION:
            thresh = self.target_score * DISTILLATION_ENTER_RATIO
            if sma_score >= thresh:
                self._transition_to_distillation(sma_score)
        elif self.phase == Phase.DISTILLATION:
            if sma_score >= self.target_score:
                self._transition_to_crystallization(sma_score)
            else:
                rollback_thresh = self.target_score * ROLLBACK_RATIO
                if sma_score < rollback_thresh:
                    self._transition_to_exploration(sma_score)

        return self.phase

    def _transition_to_distillation(self, sma_score: float):
        prev_prune, prev_night = self.current_prune_threshold, self.current_night_interval
        self.control.set_prune_threshold(DISTILLATION_PRUNE)
        self.control.set_night_interval(DISTILLATION_NIGHT)
        self.current_prune_threshold = DISTILLATION_PRUNE
        self.current_night_interval = DISTILLATION_NIGHT
        self.phase = Phase.DISTILLATION
        print(
            f"\n🔥 [AutoTuner] EXPLORATION → DISTILLATION | "
            f"SMA: {sma_score:.0f} (≥{self.target_score * DISTILLATION_ENTER_RATIO:.0f}) | "
            f"prune: {prev_prune}→{DISTILLATION_PRUNE} | night: {prev_night}→{DISTILLATION_NIGHT}"
        )

    def _transition_to_crystallization(self, sma_score: float):
        prev_prune, prev_night = self.current_prune_threshold, self.current_night_interval
        self.control.set_night_interval(0)
        self.control.set_prune_threshold(0)
        self.current_prune_threshold = 0
        self.current_night_interval = 0
        self.phase = Phase.CRYSTALLIZED
        print(
            f"\n❄️ [AutoTuner] DISTILLATION → CRYSTALLIZED | "
            f"SMA: {sma_score:.0f} (≥{self.target_score:.0f}) | "
            f"prune: {prev_prune}→0 | night: {prev_night}→0"
        )

    def _transition_to_exploration(self, sma_score: float):
        prev_prune, prev_night = self.current_prune_threshold, self.current_night_interval
        rollback_thresh = self.target_score * ROLLBACK_RATIO
        self.control.set_prune_threshold(EXPLORATION_PRUNE)
        self.control.set_night_interval(EXPLORATION_NIGHT)
        self.current_prune_threshold = EXPLORATION_PRUNE
        self.current_night_interval = EXPLORATION_NIGHT
        self.phase = Phase.EXPLORATION
        self.window.clear()
        print(
            f"\n🌱 [AutoTuner] DISTILLATION → EXPLORATION (rollback) | "
            f"SMA: {sma_score:.0f} (<{rollback_thresh:.0f}) | "
            f"prune: {prev_prune}→{EXPLORATION_PRUNE} | night: {prev_night}→{EXPLORATION_NIGHT}"
        )

    def get_punishment_modifier(self) -> float:
        """
        Множитель для shock-батчей (0.5..1.0).
        При забывании (SMA падает в DISTILLATION) возвращаем < 1.0,
        чтобы не усугублять порочный круг: забывание → reward↓ → punishment↑ → LTD → забывание↑
        """
        if self.phase != Phase.DISTILLATION:
            return 1.0
        soften_thresh = self.target_score * PUNISHMENT_SOFTEN_SMA_RATIO
        if self.last_sma < soften_thresh:
            return PUNISHMENT_SOFTEN_RATIO
        return 1.0

    def get_state_for_logging(self) -> dict:
        """Метрики для анализа: текущие параметры и пороги."""
        return {
            "phase": self.phase.name,
            "last_sma": self.last_sma,
            "target_score": self.target_score,
            "prune_threshold": self.current_prune_threshold,
            "night_interval": self.current_night_interval,
            "rollback_threshold": self.target_score * ROLLBACK_RATIO,
            "distillation_enter_threshold": self.target_score * DISTILLATION_ENTER_RATIO,
            "punishment_modifier": self.get_punishment_modifier(),
            "window_size": len(self.window),
        }
