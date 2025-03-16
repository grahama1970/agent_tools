import json


def convert_to_markdown(doc: dict) -> str:
    """Convert the structured JSON document into Markdown for LLM consumption."""
    lines = []
    page = doc.get("page", {})
    lines.append(f"# {page.get('title', 'No Title')}\n")
    lines.append(f"**URL:** {page.get('url', '')}\n")
    lines.append(f"**Summary:** {page.get('summary', '')}\n")
    lines.append("---\n")
    for section in doc.get("content_list", []):
        lines.append(f"## {section.get('section_name', 'Unnamed Section')}\n")
        for item in section.get("contents", []):
            item_type = item.get("type", "text")
            description = item.get("description", "")
            content = item.get("content", "")
            if item_type == "text":
                lines.append(content + "\n")
            elif item_type == "table":
                lines.append(f"**Table:** {description}\n")
                try:
                    table_data = json.loads(content)
                    lines.append(
                        "```json\n" + json.dumps(table_data, indent=2) + "\n```\n"
                    )
                except Exception:
                    lines.append(content + "\n")
            elif item_type == "image":
                src = item.get("src", "")
                lines.append(f"![{description}]({src})\n")
            else:
                lines.append(content + "\n")
        lines.append("---\n")
    return "\n".join(lines)
