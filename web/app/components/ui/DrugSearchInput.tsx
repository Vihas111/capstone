"use client";

import { useMemo, useRef, useState } from "react";
import { SectionLabel } from "./SectionLabel";
import styles from "./UI.module.css";

interface DrugSearchInputProps {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  drugs: string[];
  disabled?: boolean;
}

const ITEM_HEIGHT = 40; // must match .option's height in UI.module.css
const OVERSCAN = 6;

export function DrugSearchInput({
  id,
  label,
  value,
  onChange,
  drugs,
  disabled = false,
}: DrugSearchInputProps) {
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const [scrollTop, setScrollTop] = useState(0);
  const closeTimer = useRef<number>();
  const menuRef = useRef<HTMLDivElement>(null);

  const filtered = useMemo(() => {
    const query = value.trim().toLowerCase();
    if (!query) return drugs;
    return drugs.filter((drug) => drug.toLowerCase().startsWith(query));
  }, [drugs, value]);

  const visibleCount = Math.ceil(280 / ITEM_HEIGHT);
  const startIndex = Math.max(0, Math.floor(scrollTop / ITEM_HEIGHT) - OVERSCAN);
  const endIndex = Math.min(filtered.length, startIndex + visibleCount + OVERSCAN * 2);
  const visibleItems = filtered.slice(startIndex, endIndex);

  function scrollToIndex(index: number) {
    const menu = menuRef.current;
    if (!menu) return;
    const itemTop = index * ITEM_HEIGHT;
    const itemBottom = itemTop + ITEM_HEIGHT;
    if (itemTop < menu.scrollTop) {
      menu.scrollTop = itemTop;
    } else if (itemBottom > menu.scrollTop + menu.clientHeight) {
      menu.scrollTop = itemBottom - menu.clientHeight;
    }
  }

  return (
    <div className={styles.inputWrap}>
      <SectionLabel>{label}</SectionLabel>
      <input
        id={id}
        className={styles.input}
        role="combobox"
        aria-expanded={open}
        aria-controls={`${id}-listbox`}
        aria-activedescendant={open ? `${id}-opt-${activeIndex}` : undefined}
        autoComplete="off"
        disabled={disabled}
        value={value}
        onFocus={() => !disabled && setOpen(true)}
        onBlur={() => {
          closeTimer.current = window.setTimeout(() => setOpen(false), 100);
        }}
        onChange={(event) => {
          onChange(event.target.value);
          setOpen(true);
          setActiveIndex(0);
          setScrollTop(0);
        }}
        onKeyDown={(event) => {
          if (!filtered.length) return;
          if (event.key === "ArrowDown") {
            event.preventDefault();
            setOpen(true);
            setActiveIndex((prev) => {
              const next = (prev + 1) % filtered.length;
              scrollToIndex(next);
              return next;
            });
          }
          if (event.key === "ArrowUp") {
            event.preventDefault();
            setOpen(true);
            setActiveIndex((prev) => {
              const next = (prev - 1 + filtered.length) % filtered.length;
              scrollToIndex(next);
              return next;
            });
          }
          if (event.key === "Enter" && open) {
            event.preventDefault();
            onChange(filtered[activeIndex]);
            setOpen(false);
          }
          if (event.key === "Escape") {
            setOpen(false);
          }
        }}
      />
      {open && filtered.length > 0 && (
        <div
          id={`${id}-listbox`}
          className={styles.menu}
          role="listbox"
          ref={menuRef}
          onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
        >
          <div
            className={styles.menuInner}
            style={{ height: filtered.length * ITEM_HEIGHT }}
          >
            {visibleItems.map((drug, i) => {
              const index = startIndex + i;
              return (
                <button
                  key={drug}
                  id={`${id}-opt-${index}`}
                  role="option"
                  aria-selected={activeIndex === index}
                  className={`${styles.option} ${activeIndex === index ? styles.optionActive : ""}`}
                  style={{ top: index * ITEM_HEIGHT }}
                  title={drug}
                  onMouseDown={() => {
                    window.clearTimeout(closeTimer.current);
                    onChange(drug);
                    setOpen(false);
                  }}
                  onMouseEnter={() => setActiveIndex(index)}
                >
                  {drug}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
