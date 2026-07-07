from dataclasses import dataclass


COMMANDS = {
    'jarvis': 'jarvis',
    'jarvis boot up': 'jarvis boot up',
    'boot up': 'jarvis boot up',
    'enable': 'enable_hud',
    'disable': 'disable_hud',
    'disabled' : 'disable_hud',
    'scan face': 'scan_face',
    'night vision': 'night_vision',
    'night' : 'night_vision',
    'light' : 'night_vision',
    'start recording': 'start_recording',
    'stop recording': 'stop_recording',
    'take screenshot': 'take_screenshot',
    'toggle face tracking': 'toggle_face_tracking',
    'toggle diagnostics': 'toggle_diagnostics',
    'toggle hud': 'toggle_hud',
    'stop': 'stop',
}


@dataclass
class CommandAction:
    name: str
    payload: str = ''


def normalize_command(text):
    if text is None:
        return ''

    normalized = ' '.join(text.lower().strip().split())
    return normalized


def resolve_command(text):
    normalized = normalize_command(text)
    if not normalized:
        return None

    command_name = COMMANDS.get(normalized)
    if command_name:
        return CommandAction(command_name, normalized)

    if 'boot up' in normalized:
        return CommandAction('jarvis boot up', normalized)

    if 'jarvis' in normalized:
        return CommandAction('jarvis', normalized)

    if 'enable hud' in normalized or 'hud on' in normalized:
        return CommandAction('enable_hud', normalized)

    if 'disable hud' in normalized or 'hud off' in normalized:
        return CommandAction('disable_hud', normalized)

    if 'toggle hud' in normalized:
        return CommandAction('toggle_hud', normalized)

    if 'scan face' in normalized:
        return CommandAction('scan_face', normalized)

    if 'night vision' in normalized:
        return CommandAction('night_vision', normalized)

    if 'start recording' in normalized:
        return CommandAction('start_recording', normalized)

    if 'stop recording' in normalized:
        return CommandAction('stop_recording', normalized)

    if 'take screenshot' in normalized or 'screenshot' in normalized:
        return CommandAction('take_screenshot', normalized)

    if 'toggle face tracking' in normalized:
        return CommandAction('toggle_face_tracking', normalized)

    if 'toggle diagnostics' in normalized:
        return CommandAction('toggle_diagnostics', normalized)

    if normalized == 'stop' or normalized.endswith(' stop'):
        return CommandAction('stop', normalized)

    return CommandAction('unknown', normalized)