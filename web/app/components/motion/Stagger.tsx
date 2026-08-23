"use client";

import { PropsWithChildren } from "react";
import { motion } from "framer-motion";

interface StaggerProps extends PropsWithChildren {
  gap?: number;
  className?: string;
}

export function Stagger({ children, gap = 0.12, className }: StaggerProps) {
  return (
    <motion.div
      className={className}
      initial="hidden"
      animate="show"
      variants={{
        hidden: {},
        show: {
          transition: { staggerChildren: gap },
        },
      }}
    >
      {children}
    </motion.div>
  );
}
