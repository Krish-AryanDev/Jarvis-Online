from dataclasses import dataclass, field
from collections import deque
from time import time
import math
from pathlib import Path

import cv2 as cv

from overlay_utils import load_overlay, blend_overlay, overlay_on_bbox


try:
    import psutil
except ImportError:
    psutil = None


@dataclass
class ComponentLayout:
    x: int
    y: int
    scale: float = 1.0
    visible: bool = True


@dataclass
class HUDConfig:
    enabled: bool = True
    show_radar: bool = True
    show_eye_scanner: bool = True
    show_face_box: bool = True
    show_target_lock: bool = True
    show_battery: bool = True
    show_performance: bool = True
    show_scan_lines: bool = True
    show_diagnostics: bool = True
    radar: ComponentLayout = field(default_factory=lambda: ComponentLayout(120, 120, 1.0, True))
    eye_scanner: ComponentLayout = field(default_factory=lambda: ComponentLayout(1120, 110, 1.0, True))
    battery: ComponentLayout = field(default_factory=lambda: ComponentLayout(70, 640, 1.0, True))
    performance: ComponentLayout = field(default_factory=lambda: ComponentLayout(70, 700, 1.0, True))
    diagnostics: ComponentLayout = field(default_factory=lambda: ComponentLayout(70, 590, 1.0, True))
    warning: ComponentLayout = field(default_factory=lambda: ComponentLayout(460, 60, 1.0, True))


class HUDController:
    def __init__(self, config=None):
        self.config = config or HUDConfig()
        self.start_time = time()
        self.boot_start = 0
        self.notifications = deque(maxlen=8)
        self._system_metrics = {'cpu': 0.0, 'ram': 0.0}
        self.overlays = {}
        try:
            images_dir = Path('Images')
            self.overlays['radar'] = load_overlay(str(images_dir / 'hud_arc_dial.png'))
            self.overlays['eye_scanner'] = load_overlay(str(images_dir / 'hud_gauge_arc.png'))
            self.overlays['face_box'] = load_overlay(str(images_dir / 'hud_hero_frame.png'))
            self.overlays['right_eye'] = load_overlay(str(images_dir / 'hud_arc_dial.png'))
            self.overlays['target_lock'] = load_overlay(str(images_dir / 'hud_target_lock.png'))
            self.overlays['hero_frame'] = load_overlay(str(images_dir / 'hud_hero_frame.png'))
            self.overlays['telemetry'] = load_overlay(str(images_dir / 'hud_telemetry.png'))
            self.overlays['hex_lock'] = load_overlay(str(images_dir / 'hud_hex_lock.png'))
        except Exception as e:
            print(f"Failed to load HUD overlays: {e}")

    def trigger_boot_sequence(self):
        self.boot_start = time()

    def toggle(self, name, value=None):
        if hasattr(self.config, name):
            current_value = getattr(self.config, name)
            setattr(self.config, name, (not current_value) if value is None else bool(value))

    def set_component_visible(self, component_name, visible):
        component = getattr(self.config, component_name, None)
        if component is not None:
            component.visible = bool(visible)

    def move_component(self, component_name, x=None, y=None):
        component = getattr(self.config, component_name, None)
        if component is not None:
            if x is not None:
                component.x = int(x)
            if y is not None:
                component.y = int(y)

    def scale_component(self, component_name, scale):
        component = getattr(self.config, component_name, None)
        if component is not None:
            component.scale = max(0.1, float(scale))

    def notify(self, text, duration=2.5, color=(0, 255, 255)):
        self.notifications.appendleft({'text': text, 'expires_at': time() + duration, 'color': color})

    def _prune_notifications(self):
        now = time()
        self.notifications = deque([item for item in self.notifications if item['expires_at'] > now], maxlen=8)

    def _update_metrics(self):
        if psutil is None:
            return

        self._system_metrics['cpu'] = float(psutil.cpu_percent(interval=None))
        self._system_metrics['ram'] = float(psutil.virtual_memory().percent)

    def _draw_text_panel(self, frame, x, y, lines, scale=1.0, color=(0, 255, 255)):
        line_height = int(20 * scale)
        panel_height = line_height * max(1, len(lines)) + 20
        panel_width = 260
        overlay = frame.copy()
        cv.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
        cv.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        current_y = y + 22
        for line in lines:
            cv.putText(frame, line, (x + 14, current_y), cv.FONT_HERSHEY_SIMPLEX, 0.55 * scale, color, 1, cv.LINE_AA)
            current_y += line_height

    def draw_scan_lines(self, frame, tick):
        height, width = frame.shape[:2]
        sweep_y = int((tick * 120) % height)
        cv.line(frame, (0, sweep_y), (width, sweep_y), (0, 180, 180), 1, cv.LINE_AA)
        cv.line(frame, (0, sweep_y + 2), (width, sweep_y + 2), (0, 80, 80), 1, cv.LINE_AA)

    def draw_radar(self, frame, tick, layout):
        if not layout.visible or 'radar' not in self.overlays:
            return

        overlay = self.overlays['radar']
        target_size = int(180 * layout.scale)
        x = layout.x - target_size // 2
        y = layout.y - target_size // 2
        blend_overlay(frame, overlay, x, y, width=target_size, height=target_size)

    def draw_eye_scanner(self, frame, tick, layout):
        if not layout.visible or 'eye_scanner' not in self.overlays:
            return

        overlay = self.overlays['eye_scanner']
        target_size = int(90 * layout.scale)
        x = layout.x - target_size // 2
        y = layout.y - target_size // 2
        blend_overlay(frame, overlay, x, y, width=target_size, height=target_size)

    def draw_face_box(self, frame, face_box):
        if face_box is None or not self.config.show_face_box or 'face_box' not in self.overlays:
            return

        overlay_on_bbox(frame, self.overlays['face_box'], face_box, scale=1.3)

    def draw_right_eye(self, frame, right_eye_box, tick):
        if right_eye_box is None or 'right_eye' not in self.overlays:
            return

        pulse_scale = 1.0 + 0.05 * math.sin(tick * 5)
        overlay_on_bbox(frame, self.overlays['right_eye'], right_eye_box, scale=1.8 * pulse_scale)

    def draw_target_lock(self, frame, face_box, tick):
        if face_box is None or not self.config.show_target_lock or 'target_lock' not in self.overlays:
            return

        pulse_scale = 1.0 + 0.15 * math.sin(tick * 5)
        overlay_on_bbox(frame, self.overlays['target_lock'], face_box, scale=0.6 * pulse_scale)

    def draw_battery(self, frame, layout, battery_level):
        if not layout.visible or not self.config.show_battery:
            return

        level = 86 if battery_level is None else int(max(0, min(100, battery_level)))
        x, y = layout.x, layout.y
        width, height = int(170 * layout.scale), int(28 * layout.scale)
        cv.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 255), 1, cv.LINE_AA)
        fill_width = int((width - 4) * level / 100.0)
        cv.rectangle(frame, (x + 2, y + 2), (x + 2 + fill_width, y + height - 2), (0, 255, 200), -1, cv.LINE_AA)
        cv.putText(frame, f'BATT {level:02d}%', (x + 8, y - 8), cv.FONT_HERSHEY_SIMPLEX, 0.55 * layout.scale, (0, 255, 255), 1, cv.LINE_AA)

    def draw_performance(self, frame, layout, fps):
        if not layout.visible or not self.config.show_performance:
            return

        self._update_metrics()
        fps_value = 0.0 if fps is None else float(fps)
        lines = [
            f'FPS {fps_value:05.1f}',
            f'CPU {self._system_metrics["cpu"]:04.1f}%',
            f'RAM {self._system_metrics["ram"]:04.1f}%',
        ]
        self._draw_text_panel(frame, layout.x, layout.y, lines, scale=layout.scale)

    def draw_diagnostics(self, frame, layout, face_box):
        if not layout.visible or not self.config.show_diagnostics:
            return

        status = 'FACE LOCKED' if face_box is not None else 'SEARCHING FOR TARGET'
        self._draw_text_panel(frame, layout.x, layout.y, [status, 'HUD ONLINE', 'SYSTEM STABLE'], scale=layout.scale)

    def draw_notifications(self, frame, layout):
        self._prune_notifications()
        if not self.notifications:
            return

        message = self.notifications[0]
        text = message['text']
        color = message['color']
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        if 'Service' in text:
            display_text = f"[  {text.upper()}  ]"
            color = (255, 200, 0)  # Cyan
        else:
            display_text = text

        (text_width, text_height), _ = cv.getTextSize(display_text, font, font_scale, thickness)
        
        # Position in Top Right
        x = frame.shape[1] - text_width - 30
        y = text_height + 30
        
        # Background box
        padding = 10
        cv.rectangle(frame, (x - padding, y - text_height - padding), (x + text_width + padding, y + padding), (0, 0, 0), -1)
        
        # Draw text once to avoid overlaps
        cv.putText(frame, display_text, (x, y), font, font_scale, color, thickness, cv.LINE_AA)

    def draw_boot_animation(self, frame, face_box, right_eye_box, elapsed):
        if face_box is None:
            return

        x, y, w, h = face_box
        
        # Use real right eye box if available, fallback to approximation
        if right_eye_box is None:
            eye_w, eye_h = int(w * 0.25), int(h * 0.25)
            eye_x = int(x + w * 0.25)
            eye_y = int(y + h * 0.35)
            target_eye_box = (eye_x, eye_y, eye_w, eye_h)
        else:
            target_eye_box = right_eye_box

        # 1. 0 to 2 seconds: Eye scanning only
        if 'right_eye' in self.overlays and elapsed < 3.0:
            scale = min(1.8, elapsed * 2.0)
            scale += 0.05 * math.sin(elapsed * 15)  # rapid pulse
            overlay_on_bbox(frame, self.overlays['right_eye'], target_eye_box, scale=scale)

        # 2. 2 to 4 seconds: Main Hero Frame appears
        if 'hero_frame' in self.overlays and elapsed > 2.0:
            scale = min(1.3, (elapsed - 2.0) * 1.3)
            overlay_on_bbox(frame, self.overlays['hero_frame'], face_box, scale=scale)

        # 3. 3 to 5 seconds: Telemetry data pops up
        if 'telemetry' in self.overlays and elapsed > 3.0:
            alpha = min(1.0, (elapsed - 3.0))
            overlay_on_bbox(frame, self.overlays['telemetry'], face_box, scale=1.3, offset=(int(w*0.8), 0), alpha=alpha)
            overlay_on_bbox(frame, self.overlays['telemetry'], face_box, scale=1.3, offset=(-int(w*0.8), 0), alpha=alpha)

    def draw(self, frame, face_box=None, right_eye_box=None, fps=None, battery_level=None):
        if frame is None:
            return frame

        tick = time() - self.start_time

        # ALWAYS draw notifications so "At your Service sir" appears even if HUD is off
        self.draw_notifications(frame, self.config.warning)

        if not self.config.enabled:
            return frame

        elapsed_boot = time() - self.boot_start

        if elapsed_boot < 5.0:
            self.draw_boot_animation(frame, face_box, right_eye_box, elapsed_boot)
        else:
            if self.config.show_scan_lines:
                self.draw_scan_lines(frame, tick)

            if self.config.show_radar:
                self.draw_radar(frame, tick, self.config.radar)

            if self.config.show_eye_scanner:
                self.draw_eye_scanner(frame, tick, self.config.eye_scanner)

            self.draw_face_box(frame, face_box)
            self.draw_right_eye(frame, right_eye_box, tick)
            self.draw_target_lock(frame, face_box, tick)
            self.draw_battery(frame, self.config.battery, battery_level)
            self.draw_performance(frame, self.config.performance, fps)
            self.draw_diagnostics(frame, self.config.diagnostics, face_box)

        return frame