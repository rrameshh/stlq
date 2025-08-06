
import argparse
import sys
from pathlib import Path

from config import Config
from trainer import Trainer

def create_parser():
    """Simple argument parser."""
    parser = argparse.ArgumentParser(description="QAT Framework")
    
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--override", nargs="*", default=[], help="Override config (key=value)")
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    
    return parser

def main():
   
    parser = create_parser()
    args = parser.parse_args()
    
    # Load configuration
    if not Path(args.config).exists():
        print(f"Error: Config file '{args.config}' not found")
        sys.exit(1)
    
    try:
        print(f"Loading config: {args.config}")
        config = Config.from_yaml(args.config)
        
        # Apply overrides
        if args.override:
            config.apply_overrides(args.override)
        
        # Print config summary
        config.print_summary()
        
        # Save config to output directory
        config.save_yaml(Path(config.system.work_dir) / "config.yaml")
        
        if args.dry_run:
            print("Dry run complete.")
            return
        
        # Train
        trainer = Trainer(config)
        best_metric = trainer.train()
        
        print(f"Training completed! Best metric: {best_metric:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()