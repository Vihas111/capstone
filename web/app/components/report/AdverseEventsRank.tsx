"use client";

import { motion } from "framer-motion";
import { PredictResponse } from "../../lib/api";
import { SectionLabel } from "../ui/SectionLabel";
import styles from "./Report.module.css";

export function AdverseEventsRank({ result }: { result: PredictResponse }) {
  return (
    <section className={styles.section}>
      <SectionLabel>Pairwise Interaction-Risk Models</SectionLabel>
      {result.predicted_ades.map((item, index) => {
        const width = `${Math.max(2, Math.min(100, item.score * 100))}%`;
        return (
          <div key={item.ade} className={styles.adeRow} title={`Probability ${item.score.toFixed(3)}`}>
            <span className={styles.adeRank}>{String(index + 1).padStart(2, "0")}</span>
            <div>
              <div className={styles.adeName}>{item.ade}</div>
              <div className={styles.adeTrack}>
                <motion.div
                  className={styles.adeFill}
                  initial={{ width: 0 }}
                  animate={{ width }}
                  transition={{ duration: 0.9, ease: "easeOut", delay: index * 0.05 }}
                />
              </div>
            </div>
            <span className={styles.adeScore}>{item.score.toFixed(3)}</span>
          </div>
        );
      })}
    </section>
  );
}
