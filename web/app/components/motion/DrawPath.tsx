"use client";

import { motion } from "framer-motion";
import { transitions } from "../../lib/motion";

interface DrawPathProps {
  path: string;
  className?: string;
}

export function DrawPath({ path, className }: DrawPathProps) {
  return (
    <svg className={className} viewBox="0 0 10 100" preserveAspectRatio="none" aria-hidden>
      <motion.path
        d={path}
        stroke="currentColor"
        strokeWidth={1.25}
        fill="none"
        initial={{ pathLength: 0, opacity: 0.4 }}
        animate={{ pathLength: 1, opacity: 1 }}
        transition={transitions.draw}
      />
    </svg>
  );
}
