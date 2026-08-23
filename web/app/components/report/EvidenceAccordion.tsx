"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { PredictResponse } from "../../lib/api";
import { SectionLabel } from "../ui/SectionLabel";
import styles from "./Report.module.css";

export function EvidenceAccordion({ result }: { result: PredictResponse }) {
  const [openIndex, setOpenIndex] = useState(0);
  return (
    <section className={styles.section}>
      <SectionLabel>DrugBank Mechanism Evidence</SectionLabel>
      {result.grounding.map((item, index) => {
        const isOpen = openIndex === index;
        return (
          <div key={`${item.drug}-${item.section}-${index}`} className={styles.evidenceItem}>
            <button className={styles.evidenceButton} onClick={() => setOpenIndex(isOpen ? -1 : index)}>
              <span>
                DrugBank <span className={styles.evidenceMeta}>{item.drug} · {item.section}</span>
              </span>
              <motion.span animate={{ rotate: isOpen ? 180 : 0 }}>↓</motion.span>
            </button>
            <motion.div
              initial={false}
              animate={{ height: isOpen ? "auto" : 0, opacity: isOpen ? 1 : 0 }}
              style={{ overflow: "hidden" }}
            >
              <div className={styles.evidenceText}>{item.text}</div>
            </motion.div>
          </div>
        );
      })}
    </section>
  );
}
