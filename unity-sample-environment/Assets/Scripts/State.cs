using UnityEngine;
using System.Collections;

namespace MLPlayer {
	public class State {
		public float reward;
		public bool endEpisode;
		public byte[][] image;
		public byte[][] depth;
		public float[] distances = new float[32];
		public void Clear() {
			reward = 0;
			endEpisode = false;
			image = null;
			depth = null;
			distances = new float[32];
		}
	}
}