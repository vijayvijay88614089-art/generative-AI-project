from .data_loader import profile


def build_metadata(frame) -> dict:
    details = profile(frame)
    details["statistical_summary"] = frame.describe(include="all").replace({float("nan"): None}).to_dict()
    return details
