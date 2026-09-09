try:
    import yaml
    def _parse_yaml(f):
        return yaml.load(f, Loader=yaml.FullLoader)
except ImportError:
    import json
    def _parse_yaml(f):
        data = {}
        current_section = None
        for line in f:
            raw_line = line.rstrip()
            line_str = raw_line.strip()
            if not line_str or line_str.startswith("#"):
                continue
            if ":" in line_str:
                k, v = line_str.split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.startswith('"') and v.endswith('"') and len(v) >= 2:
                    v = v[1:-1]
                elif v.startswith("'") and v.endswith("'") and len(v) >= 2:
                    v = v[1:-1]
                elif v.lower() == "true":
                    v = True
                elif v.lower() == "false":
                    v = False
                elif v.isdigit():
                    v = int(v)
                elif v.startswith("[") and v.endswith("]"):
                    try:
                        v = json.loads(v)
                    except Exception:
                        pass
                
                # Check indentation (nested under current_section)
                if raw_line.startswith("  ") and current_section:
                    if not isinstance(data.get(current_section), dict):
                        data[current_section] = {}
                    data[current_section][k] = v
                else:
                    if v == "" or v is None:
                        current_section = k
                        data[k] = {}
                    else:
                        current_section = None
                        data[k] = v
        return data


def load_rag_config(config_path: str = None, encoding: str = "utf-8"):
    from .path_tool import get_abs_path
    if config_path is None:
        config_path = get_abs_path("config/rag.yml")
    with open(config_path, "r", encoding=encoding) as f:
        return _parse_yaml(f)


def load_chroma_config(config_path: str = None, encoding: str = "utf-8"):
    from .path_tool import get_abs_path
    if config_path is None:
        config_path = get_abs_path("config/chroma.yml")
    with open(config_path, "r", encoding=encoding) as f:
        return _parse_yaml(f)


def load_prompts_config(config_path: str = None, encoding: str = "utf-8"):
    from .path_tool import get_abs_path
    if config_path is None:
        config_path = get_abs_path("config/prompts.yml")
    with open(config_path, "r", encoding=encoding) as f:
        return _parse_yaml(f)


def load_agent_config(config_path: str = None, encoding: str = "utf-8"):
    from .path_tool import get_abs_path
    if config_path is None:
        config_path = get_abs_path("config/agent.yml")
    with open(config_path, "r", encoding=encoding) as f:
        return _parse_yaml(f)


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()

