"use client";

import { motion } from "framer-motion";
import { transitions } from "../../lib/motion";
import styles from "./UI.module.css";

interface AnalyzeButtonProps {
  disabled: boolean;
  onClick: () => void;
}

export function AnalyzeButton({ disabled, onClick }: AnalyzeButtonProps) {
  return (
    <motion.button
      type="button"
      className={styles.analyzeButton}
      onClick={onClick}
      disabled={disabled}
      whileHover={disabled ? undefined : { scale: 1.015 }}
      whileTap={disabled ? undefined : { scale: 0.985 }}
      transition={transitions.micro}
    >
      Analyze
    </motion.button>
  );
}
