import { motion } from "framer-motion";
import { DOMAIN_OPTIONS } from "../data/domains";
import type { DomainKey } from "../types/blockchain";

interface DomainSwitcherProps {
  value: DomainKey;
  onChange: (domain: DomainKey) => void;
  disabled?: boolean;
}

export function DomainSwitcher({ value, onChange, disabled }: DomainSwitcherProps) {
  return (
    <div className="flex flex-wrap gap-3">
      {DOMAIN_OPTIONS.map((option) => {
        const Icon = option.icon;
        const isActive = option.key === value;

        return (
          <motion.button
            key={option.key}
            type="button"
            whileHover={{ scale: disabled ? 1 : 1.02 }}
            whileTap={{ scale: disabled ? 1 : 0.98 }}
            disabled={disabled}
            onClick={() => onChange(option.key)}
            className={`group relative flex flex-1 min-w-[180px] items-start gap-3 rounded-xl border border-slate-800/80 bg-slate-900/60 p-4 text-left shadow-glow transition-colors ${
              isActive
                ? "border-slate-500/80 bg-slate-900" 
                : "hover:border-slate-600/80 hover:bg-slate-900/80"
            }`}
          >
            <div
              className={`flex h-10 w-10 items-center justify-center rounded-lg bg-gradient-to-br ${option.accent}`}
            >
              <Icon className="h-5 w-5 text-slate-950" />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-100">{option.label}</p>
              <p className="text-xs text-slate-400">{option.description}</p>
            </div>
            {isActive && (
              <motion.span
                layoutId="domain-pill"
                className="absolute inset-0 -z-10 rounded-xl border border-slate-400/40"
                transition={{ type: "spring", stiffness: 180, damping: 18 }}
              />
            )}
          </motion.button>
        );
      })}
    </div>
  );
}
