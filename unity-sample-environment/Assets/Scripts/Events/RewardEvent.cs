using UnityEngine;
using System.Collections;

namespace MLPlayer {
	public class RewardEvent : MonoBehaviour {
		[SerializeField] float reward;
		[SerializeField] bool deactiveWhenCollision;

		void OnEvent(GameObject other) {
			if (other.tag == Defs.PLAYER_TAG) {
				other.GetComponent<Agent> ().AddReward (reward);
				Debug.Log ("reward:" + reward.ToString ());
				if (deactiveWhenCollision) {
					gameObject.SetActive (false);
				}
			}
		}

		void OnTriggerEnter(Collider other) {
			OnEvent (other.gameObject);
		}

		void OnCollisionEnter(Collision collision) {
			OnEvent (collision.gameObject);
		}

		void OnCollisionStay(Collision collision) {
			OnEvent (collision.gameObject);
		}

	}
}