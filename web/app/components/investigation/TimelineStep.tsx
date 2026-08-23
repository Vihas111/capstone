"use client";

import { motion } from "framer-motion";
import { DrawPath } from "../motion/DrawPath";
import styles from "./Investigation.module.css";

interface TimelineStepProps {
  label: string;
  active: boolean;
  completed: boolean;
  isLast: boolean;
}

export function TimelineStep({ label, active, completed, isLast }: TimelineStepProps) {
  const dotClass = completed
    ? `${styles.dot} ${styles.dotCompleted}`
    : active
      ? `${styles.dot} ${styles.dotActive}`
      : styles.dot;

  const labelClass = completed
    ? `${styles.label} ${styles.labelCompleted}`
    : active
      ? `${styles.label} ${styles.labelActive}`
      : styles.label;

  return (
    <div>
      <div className={styles.step}>
        <motion.span
          className={dotClass}
          animate={active ? { scale: [1, 1.12, 1] } : { scale: 1 }}
          transition={{ duration: 1.2, repeat: active ? Infinity : 0 }}
        />
        <p className={labelClass}>{label}</p>
      </div>
      {!isLast && completed && <DrawPath path="M5 0 L5 100" className={styles.line} />}
    </div>
  );
}
