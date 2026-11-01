import React, { useState, useId } from "react";
import { HelpCircle, ChevronDown, ChevronRight, Minus, Plus } from "lucide-react";

interface HelpTooltipProps {
  text: string;
}

export const HelpTooltip: React.FC<HelpTooltipProps> = ({ text }) => {
  const [open, setOpen] = useState(false);

  return (
    <span className="relative inline-flex items-center">
      <button
        type="button"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        aria-label={text}
        className="text-[#fafafa]/50 hover:text-[#fafafa] transition-colors focus-visible:outline-2 focus-visible:outline-[#ff4b4b] rounded-sm p-0.5 cursor-help"
      >
        <HelpCircle className="w-3.5 h-3.5" />
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute bottom-full right-0 mb-2 z-50 w-60 p-2.5 text-xs leading-relaxed font-normal text-white bg-[#1e222d] border border-white/15 rounded-md shadow-xl pointer-events-none"
        >
          {text}
        </span>
      )}
    </span>
  );
};

interface FieldLabelRowProps {
  htmlFor?: string;
  label: string;
  tooltip?: string;
  actionLabel?: string;
  onAction?: () => void;
}

export const FieldLabelRow: React.FC<FieldLabelRowProps> = ({
  htmlFor,
  label,
  tooltip,
  actionLabel,
  onAction,
}) => {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-x-2 gap-y-0.5 mb-1.5">
      <div className="flex items-center gap-1.5 min-w-0">
        <label
          htmlFor={htmlFor}
          className="text-[14px] leading-5 font-normal text-[#fafafa] tracking-[0.01em]"
        >
          {label}
        </label>
        {actionLabel && onAction && (
          <button
            type="button"
            onClick={onAction}
            className="text-[12.8px] leading-5 text-[#58a6ff] hover:underline underline-offset-2 whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-2 focus-visible:outline-[#ff4b4b] rounded-xs"
          >
            {actionLabel}
          </button>
        )}
      </div>
      {tooltip && <HelpTooltip text={tooltip} />}
    </div>
  );
};

interface SelectFieldProps {
  label?: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
  tooltip?: string;
  actionLabel?: string;
  onAction?: () => void;
  disabled?: boolean;
  ariaLabel?: string;
}

export const SelectField: React.FC<SelectFieldProps> = ({
  label,
  value,
  options,
  onChange,
  tooltip,
  actionLabel,
  onAction,
  disabled = false,
  ariaLabel,
}) => {
  const id = useId();
  return (
    <div className="w-full">
      {label && (
        <FieldLabelRow
          htmlFor={id}
          label={label}
          tooltip={tooltip}
          actionLabel={actionLabel}
          onAction={onAction}
        />
      )}
      <div className="relative">
        <select
          id={id}
          aria-label={ariaLabel || label}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value)}
          className="w-full min-h-[40px] py-2 pl-3 pr-9 appearance-none rounded-lg bg-[#262730] border border-white/15 hover:border-white/35 focus:border-[#ff4b4b] focus:ring-1 focus:ring-[#ff4b4b] outline-none text-[14px] text-[#fafafa] transition-colors cursor-pointer disabled:opacity-55 disabled:cursor-not-allowed"
        >
          {options.map((opt) => (
            <option key={opt} value={opt} className="bg-[#262730] text-[#fafafa]">
              {opt}
            </option>
          ))}
        </select>
        <ChevronDown className="w-4 h-4 text-[#fafafa]/65 pointer-events-none absolute right-3 top-1/2 -translate-y-1/2" />
      </div>
    </div>
  );
};

interface SliderFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  formatValue: (val: number) => string;
  onChange: (val: number) => void;
  tooltip?: string;
  disabled?: boolean;
}

export const SliderField: React.FC<SliderFieldProps> = ({
  label,
  value,
  min,
  max,
  step,
  formatValue,
  onChange,
  tooltip,
  disabled = false,
}) => {
  const id = useId();
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100));

  return (
    <div className={`w-full ${disabled ? "opacity-50 pointer-events-none" : ""}`}>
      <div className="flex items-center justify-between gap-2 mb-1">
        <label
          htmlFor={id}
          className="text-[14px] leading-5 text-[#fafafa] tracking-[0.01em]"
        >
          {label}
        </label>
        <div className="flex items-center gap-1.5">
          <span className="font-mono tabular-nums text-[12.5px] font-medium text-[#ff4b4b]">
            {formatValue(value)}
          </span>
          {tooltip && <HelpTooltip text={tooltip} />}
        </div>
      </div>
      <div className="pt-1 pb-0.5">
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          style={{
            background: `linear-gradient(to right, #ff4b4b 0%, #ff4b4b ${pct}%, rgba(151, 166, 195, 0.3) ${pct}%, rgba(151, 166, 195, 0.3) 100%)`,
          }}
          className="mpt-slider"
        />
        <div className="flex justify-between text-[11px] font-mono tabular-nums text-[#fafafa]/50 mt-1">
          <span>{formatValue(min)}</span>
          <span>{formatValue(max)}</span>
        </div>
      </div>
    </div>
  );
};

interface NumberStepperProps {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (val: number) => void;
  tooltip?: string;
}

export const NumberStepper: React.FC<NumberStepperProps> = ({
  label,
  value,
  min,
  max,
  onChange,
  tooltip,
}) => {
  const id = useId();
  return (
    <div className="w-full">
      <FieldLabelRow htmlFor={id} label={label} tooltip={tooltip} />
      <div className="flex items-center h-[40px] rounded-lg bg-[#262730] border border-white/15 focus-within:border-[#ff4b4b] focus-within:ring-1 focus-within:ring-[#ff4b4b] overflow-hidden">
        <input
          id={id}
          type="number"
          min={min}
          max={max}
          value={value}
          onChange={(e) => {
            const parsed = parseInt(e.target.value, 10);
            if (!Number.isNaN(parsed)) {
              onChange(Math.max(min, Math.min(max, parsed)));
            }
          }}
          className="w-full h-full px-3 bg-transparent outline-none font-mono tabular-nums text-[14px] text-[#fafafa]"
        />
        <button
          type="button"
          aria-label="Decrease"
          disabled={value <= min}
          onClick={() => onChange(Math.max(min, value - 1))}
          className="h-full px-2.5 flex items-center justify-center text-[#fafafa]/75 hover:bg-white/10 disabled:opacity-35 transition-colors cursor-pointer"
        >
          <Minus className="w-3.5 h-3.5" />
        </button>
        <button
          type="button"
          aria-label="Increase"
          disabled={value >= max}
          onClick={() => onChange(Math.min(max, value + 1))}
          className="h-full px-2.5 flex items-center justify-center text-[#fafafa]/75 hover:bg-white/10 disabled:opacity-35 transition-colors cursor-pointer"
        >
          <Plus className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};

interface CheckboxFieldProps {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  tooltip?: string;
  disabled?: boolean;
}

export const CheckboxField: React.FC<CheckboxFieldProps> = ({
  label,
  checked,
  onChange,
  tooltip,
  disabled = false,
}) => {
  const id = useId();
  return (
    <div className={`flex items-center justify-between gap-2 py-0.5 ${disabled ? "opacity-50" : ""}`}>
      <label
        htmlFor={id}
        className="inline-flex items-center gap-2.5 cursor-pointer select-none min-h-[28px]"
      >
        <input
          id={id}
          type="checkbox"
          checked={checked}
          disabled={disabled}
          onChange={(e) => onChange(e.target.checked)}
          className="w-4 h-4 rounded border-white/30 accent-[#ff4b4b] cursor-pointer focus-visible:outline-2 focus-visible:outline-[#ff4b4b]"
        />
        <span className="text-[14px] leading-5 text-[#fafafa] tracking-[0.01em]">
          {label}
        </span>
      </label>
      {tooltip && <HelpTooltip text={tooltip} />}
    </div>
  );
};

interface SegmentedControlProps<T extends string> {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (val: T) => void;
}

export function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
}: SegmentedControlProps<T>) {
  return (
    <div className="w-full">
      <div className="text-[14px] leading-5 text-[#fafafa] tracking-[0.01em] mb-1.5">
        {label}
      </div>
      <div
        role="radiogroup"
        aria-label={label}
        className="flex items-center p-1 rounded-lg bg-[#262730] border border-white/12 gap-1"
      >
        {options.map((opt) => {
          const active = value === opt;
          return (
            <button
              key={opt}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => onChange(opt)}
              className={`flex-1 min-w-0 py-1.5 px-2 rounded-md text-[13px] font-medium whitespace-nowrap truncate transition-colors cursor-pointer ${
                active
                  ? "bg-[#161922] text-[#ff4b4b] shadow-xs border border-[#ff4b4b]/35"
                  : "text-[#fafafa]/75 hover:text-[#fafafa]"
              }`}
            >
              {opt}
            </button>
          );
        })}
      </div>
    </div>
  );
}

interface ColorPickerFieldProps {
  label: string;
  value: string;
  onChange: (hex: string) => void;
  disabled?: boolean;
}

export const ColorPickerField: React.FC<ColorPickerFieldProps> = ({
  label,
  value,
  onChange,
  disabled = false,
}) => {
  const id = useId();
  return (
    <div className={disabled ? "opacity-50 pointer-events-none" : ""}>
      <label
        htmlFor={id}
        className="block text-[14px] leading-5 text-[#fafafa] tracking-[0.01em] mb-1.5 whitespace-nowrap"
      >
        {label}
      </label>
      <div className="inline-flex items-center gap-2 h-[38px] px-2 rounded-lg bg-[#262730] border border-white/15">
        <input
          id={id}
          type="color"
          value={value}
          disabled={disabled}
          onChange={(e) => onChange(e.target.value.toUpperCase())}
          className="w-6 h-6 rounded border-0 bg-transparent cursor-pointer p-0"
        />
        <span className="font-mono tabular-nums text-[12px] text-[#fafafa]/85">
          {value.toUpperCase()}
        </span>
      </div>
    </div>
  );
};

interface ExpanderProps {
  title: string;
  variant?: "subtle" | "bordered";
  defaultOpen?: boolean;
  children: React.ReactNode;
}

export const Expander: React.FC<ExpanderProps> = ({
  title,
  variant = "subtle",
  defaultOpen = false,
  children,
}) => {
  const [open, setOpen] = useState(defaultOpen);

  if (variant === "subtle") {
    return (
      <div className="w-full rounded">
        <button
          type="button"
          aria-expanded={open}
          onClick={() => setOpen((v) => !v)}
          className={`w-full min-h-[36px] px-2 py-1.5 flex items-center gap-1.5 text-left text-[14px] font-medium text-[#fafafa] transition-colors cursor-pointer rounded-t ${
            open
              ? "bg-white/[0.06] opacity-100"
              : "opacity-80 hover:opacity-100 hover:bg-white/[0.06] rounded-b"
          }`}
        >
          {open ? (
            <ChevronDown className="w-4 h-4 shrink-0" />
          ) : (
            <ChevronRight className="w-4 h-4 shrink-0" />
          )}
          <span className="whitespace-nowrap truncate">{title}</span>
        </button>
        {open && (
          <div className="px-3.5 pt-2.5 pb-3.5 bg-white/[0.045] rounded-b space-y-3.5">
            {children}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="w-full rounded-lg border border-white/14 bg-[#0e1117]">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
        className="w-full min-h-[44px] px-4 py-2.5 flex items-center justify-between text-left text-[14px] font-medium text-[#fafafa] hover:bg-white/[0.03] transition-colors cursor-pointer rounded-lg"
      >
        <span className="flex items-center gap-2 whitespace-nowrap">
          {open ? (
            <ChevronDown className="w-4 h-4 shrink-0 text-[#fafafa]/70" />
          ) : (
            <ChevronRight className="w-4 h-4 shrink-0 text-[#fafafa]/70" />
          )}
          {title}
        </span>
      </button>
      {open && (
        <div className="px-4 pb-4 pt-2 border-t border-white/10">
          {children}
        </div>
      )}
    </div>
  );
};
