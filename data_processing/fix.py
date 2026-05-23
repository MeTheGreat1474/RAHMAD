content = open(r'C:\Users\User\Documents\Programming\fyp\RAHMAD\data_processing\train_early_predictor.py', 'r', encoding='utf-8').read()

old = '    # \u2500\u2500 Save model artifacts \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n    model.clip.save_pretrained(output_dir / "lora_adapter")\n    torch.save(model.mlp.state_dict(), output_dir / "mlp_head.pt")\n    processor.save_pretrained(output_dir / "processor")\n\n    log.info(f"Artifacts saved \u2192 {output_dir}")\n    log.info("Training complete.")'

new = '    # \u2500\u2500 Save model artifacts (DirectML workaround) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n    log.info("Rebuilding model on CPU for saving...")\n    from transformers import CLIPModel as _CLIPModel\n    _cpu_clip = _CLIPModel.from_pretrained(args.clip_model)\n    _lora_cfg2 = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)\n    _cpu_model = EarlyPredictor(_cpu_clip, _lora_cfg2)\n    _saved_ckpt = torch.load(output_dir / "best_checkpoint.pt", map_location="cpu")\n    _cpu_model.load_state_dict(_saved_ckpt["model_state_dict"])\n    _cpu_model.eval()\n    _cpu_model.clip.save_pretrained(output_dir / "lora_adapter")\n    torch.save(_cpu_model.mlp.state_dict(), output_dir / "mlp_head.pt")\n    processor.save_pretrained(output_dir / "processor")\n\n    log.info(f"Artifacts saved to {output_dir}")\n    log.info("Training complete.")'

if old in content:
    content = content.replace(old, new)
    open(r'C:\Users\User\Documents\Programming\fyp\RAHMAD\data_processing\train_early_predictor.py', 'w', encoding='utf-8').write(content)
    print('Patched successfully')
else:
    print('Pattern not found - trying line-based replace')
    lines = content.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and 'Save model artifacts' in lines[i] and 'save_pretrained' in lines[i+1]:
            new_lines.append('    # Save model artifacts (DirectML workaround)')
            new_lines.append('    log.info("Rebuilding model on CPU for saving...")')
            new_lines.append('    from transformers import CLIPModel as _CLIPModel')
            new_lines.append('    _cpu_clip = _CLIPModel.from_pretrained(args.clip_model)')
            new_lines.append('    _lora_cfg2 = build_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)')
            new_lines.append('    _cpu_model = EarlyPredictor(_cpu_clip, _lora_cfg2)')
            new_lines.append('    _saved_ckpt = torch.load(output_dir / "best_checkpoint.pt", map_location="cpu")')
            new_lines.append('    _cpu_model.load_state_dict(_saved_ckpt["model_state_dict"])')
            new_lines.append('    _cpu_model.eval()')
            new_lines.append('    _cpu_model.clip.save_pretrained(output_dir / "lora_adapter")')
            new_lines.append('    torch.save(_cpu_model.mlp.state_dict(), output_dir / "mlp_head.pt")')
            new_lines.append('    processor.save_pretrained(output_dir / "processor")')
            new_lines.append('    log.info(f"Artifacts saved to {output_dir}")')
            new_lines.append('    log.info("Training complete.")')
            # skip old lines until Training complete
            while i < len(lines) and 'Training complete' not in lines[i]:
                i += 1
            i += 1  # skip Training complete line too
        else:
            new_lines.append(lines[i])
            i += 1
    open(r'C:\Users\User\Documents\Programming\fyp\RAHMAD\data_processing\train_early_predictor.py', 'w', encoding='utf-8').write('\n'.join(new_lines))
    print('Patched via line-based replace')
