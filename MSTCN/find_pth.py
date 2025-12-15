from pathlib import Path
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

root_dir = Path('./model')

# 递归查找所有 event 文件
event_files = list(root_dir.rglob("events.out.tfevents.*"))
print(f"Found {len(event_files)} TensorBoard event files.")

results = []

for event_file in sorted(event_files):
    print(f"\n🔍 Processing: {event_file.relative_to(root_dir)}")
    try:
        loader = EventFileLoader(str(event_file))
        graph_def = None
        for event in loader.Load():
            if event.HasField("graph_def") and event.graph_def:
                graph_def = event.graph_def
                break  # 通常只有一个 graph

        if graph_def is None:
            print("  ⚠️  No graph_def found in event file.")
            results.append((str(event_file), []))
            continue

        # 将 graph_def 转为可搜索的字符串
        graph_str = graph_def.decode('utf-8', errors='ignore').lower()
        stages = []
        if 'squeeze2' in graph_str:
            stages.append('Squeeze2Stage')
        if 'squeeze1' in graph_str:
            stages.append('Squeeze1Stage')
        if 'squeeze0' in graph_str:
            stages.append('Squeeze0Stage')

        if stages:
            print(f"  ✅ Detected: {', '.join(stages)}")
        else:
            # 可选：打印一小段用于调试
            print("  ℹ️  No keywords found. Sample snippet:")
            print("      " + graph_str[:300].replace('\n', ' ') + " ...")

        results.append((str(event_file), stages))

    except Exception as e:
        print(f"  ❌ Error: {e}")
        results.append((str(event_file), None))

# 汇总
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for path, stages in results:
    label = (
        "(error)" if stages is None else
        ", ".join(stages) if stages else
        "(none)"
    )
    print(f"{path} → {label}")

# squeeze 2 stage
# model\model\v2-2-128\2025-08-11-12-23-37\tb_logs2025-08-11-12-23-37\events.out.tfevents.1754886217.localhost.localdomain.1555063.0 → Squeeze2Stage
# squeeze 0 stage
# model\v2-1-128\2025-08-09-18-16-48\tb_logs2025-08-09-18-16-48\events.out.tfevents.1754734608.localhost.localdomain.3479651.0 → Squeeze0Stage
# squeeze 1 stage 32
# model\model\v2-2-32\2025-08-09-22-05-52\tb_logs2025-08-09-22-05-52\events.out.tfevents.1754748352.localhost.localdomain.3973973.0 → Squeeze1Stage
# squeeze 1 stage 128:
# our default result.
# squeeze 1 stage 256
# model\model\v2-2-256\2025-08-12-11-11-04\tb_logs2025-08-12-11-11-04\events.out.tfevents.1754968264.localhost.localdomain.2672777.0 → Squeeze1Stage
