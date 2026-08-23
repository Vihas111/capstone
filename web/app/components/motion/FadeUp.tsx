"use client";

import { HTMLMotionProps, motion } from "framer-motion";
import { transitions } from "../../lib/motion";

export function FadeUp(props: HTMLMotionProps<"div">) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: 8 }}
      transition={transitions.enter}
      {...props}
    />
  );
}
